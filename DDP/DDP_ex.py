import os
import random
import glob

import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from sklearn.metrics import f1_score # f1 score 


import timm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import warnings
from sklearn.model_selection import StratifiedKFold

# ======================== Setup & Cleanup ========================

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # dist.init_process_group("gloo", rank=rank, world_size=world_size) # windows
    dist.init_process_group("nccl", rank=rank, world_size=world_size) # windows
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# ======================== Configuration ========================

CFG = {
    'IMG_SIZE': 448,
    'EPOCHS': 30,
    'LEARNING_RATE': 1e-5,
    'BATCH_SIZE': 8,
    'SEED': 1042,
    'NUM_CLASSES': 7,
    'MODEL_NAME': 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',
    'SAVE_PATH': './best_model.pth'
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# ======================== Dataset ========================

class CustomDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        label = self.labels[idx]
        return image, label

# ======================== Model Wrapper ========================

class BaseModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)
        self.classifier = nn.Linear(1000, num_classes)
        self.criterion = FocalLoss(alpha=1, gamma=2)
        # self.f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def compute_loss(self, batch):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        return loss, outputs, labels


# ======================== focal loss ================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)

        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        
# ======================== Train & Validation ========================

def validate(model, val_loader, rank , epoch):
    model.eval()
    
    val_loss = []
    preds, true_labels = [], []
    
    criterion = FocalLoss(alpha=1,gamma=2)

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"[GPU {rank}] validation"):
            labels = labels.type(torch.LongTensor)
            images = images.to(rank, non_blocking=True)
            labels = labels.to(rank, non_blocking=True)

            pred = model(images)
            loss = criterion(pred, labels)
            
            val_loss.append(loss.item())
            
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += labels.detach().cpu().numpy().tolist()
            
        _val_loss = np.mean(val_loss)
        _val_score = f1_score(true_labels, preds, average='macro')

        # 예측 결과와 실제 값을 pandas DataFrame으로 생성
        # Calculate F1 score for each label
        f1_per_class = f1_score(true_labels, preds, average=None)

        # Create a DataFrame for label-wise F1 scores
        f1_df = pd.DataFrame({
            'Label': le.classes_,
            'F1 Score': f1_per_class
        })

        # Save the F1 scores to a CSV file
        f1_df.to_csv(f"./results_folder/{epoch}_f1_scores_per_label_2.csv", index=False)
        print(f"F1 scores per label saved to {epoch}_f1_scores_per_label.csv")

        # # Save the validation results
        # results_df = pd.DataFrame({
        #     'gt': true_labels,
        #     'pred': preds
        # })
        
        # results_df.to_csv(f"./results_folder/validation_results_2_{epoch}.csv", index=False)
        # print(f"Validation results saved to validation_results_{epoch}.csv")
        
    return _val_loss, _val_score

# ======================== Training ========================
le = None 

def train(rank, world_size):
    global le 
    
    setup(rank, world_size)
    seed_everything(CFG['SEED'])

    # === 데이터 준비 ===
    all_img_list = glob.glob('../data/train/*/*')
    df = pd.DataFrame({'img_path': all_img_list})
    df['rock_type'] = df['img_path'].apply(lambda x: x.split('/')[3])
    le = preprocessing.LabelEncoder()
    df['rock_type'] = le.fit_transform(df['rock_type'])
    # df = df.sample(1000 , random_state=42)
    # train_df, val_df = train_test_split(df, test_size=0.3, stratify=df['rock_type'], random_state=CFG['SEED'])

    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=CFG['SEED'])
    
    target_fold = 2
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['rock_type'])):
        if fold == target_fold:
            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)
            break

    # === transform ===
    model_tmp = timm.create_model(CFG['MODEL_NAME'], pretrained=True)
    config = timm.data.resolve_data_config({}, model=model_tmp)
    train_transform = A.Compose([
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE'] ,interpolation=cv2.INTER_CUBIC),
        A.Affine(rotate=(-360,360),shear={"x": (-10, 10), "y": (-10, 10)}, border_mode = 1,p = 1 ),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p= 0.5),
        A.Morphological(scale = (1,3), operation="erosion",p = 0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p = 0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CoarseDropout(num_holes_range=(3, 5) , p = 0.5 ),
        A.RandomResizedCrop( size = (CFG['IMG_SIZE'], CFG['IMG_SIZE']), scale = (0.7,1),ratio=(0.75, 1.33), p=0.5),  # Random zoom effect
        A.Normalize(mean=config['mean'], std=config['std']),
        ToTensorV2()
    ])
    
    
    test_transform = A.Compose([
        # PadSquare(value=(0, 0, 0)),
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE'] ,interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=config['mean'], std=config['std']),
        ToTensorV2()
    ])
    

    # === dataset & dataloader ===
    train_dataset = CustomDataset(train_df['img_path'].values, train_df['rock_type'].values, transform=train_transform)
    val_dataset = CustomDataset(val_df['img_path'].values, val_df['rock_type'].values, transform=test_transform)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], sampler=train_sampler, num_workers=4,pin_memory=True,prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], sampler=val_sampler, num_workers=4,pin_memory=True,prefetch_factor=2)

    # === 모델, 옵티마이저, DDP 래핑 ===
    model = BaseModel(CFG['MODEL_NAME'], CFG['NUM_CLASSES']).to(rank)
    model = DDP(model, device_ids=[rank])
    optimizer = optimizer = torch.optim.AdamW(params = model.parameters(), lr = CFG["LEARNING_RATE"], weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=2, threshold_mode='abs', min_lr=1e-8)
    
    
    patience = 5
    for epoch in range(CFG['EPOCHS']):
        # print("==============================================================================")
        model.train()

        best_score = 0
        early_stop_counter = 0
        best_model = None
        save_path = f"best_model_{CFG['MODEL_NAME']}.pth"
        train_loss = []
        
        for batch in tqdm(train_loader, desc=f"[GPU {rank}] Epoch {epoch+1}/{CFG['EPOCHS']}"):
            images, labels = batch
            labels = labels.type(torch.LongTensor)
            images = images.to(rank, non_blocking=True)
            labels = labels.to(rank, non_blocking=True)

            optimizer.zero_grad()
            loss, output, label = model.module.compute_loss((images, labels))
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        
        _val_loss, _val_score = validate(model, val_loader, rank, epoch+1)
        _train_loss = np.mean(train_loss)

        
        # torch.save(model.state_dict(), f"./results_models/{epoch+1}_{CFG['MODEL_NAME']}_{_val_loss:.5f}.pth")
        if scheduler is not None:
            scheduler.step(_val_score)
        
        if rank == 0:
            print(f"[Epoch {epoch+1}] Train Loss: {_train_loss:.4f} | "
                  f"Val Loss: {_val_loss:.4f} | Val F1: {_val_score:.4f}")
            
            if best_score < _val_score:
                early_stop_counter = 0
                best_score = _val_score
                best_model = model
                # 모델 가중치 저장
                torch.save(best_model.state_dict(), f"./results_models/DDP_{epoch+1}_best_{CFG['MODEL_NAME']}.pth")
                print(f"Best model saved (epoch {epoch+1}, F1={_val_score:.4f}) → {save_path}")
            else:
                early_stop_counter += 1
                print(f"No improvement for {early_stop_counter} epoch(s)")

                if early_stop_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
    cleanup()

# ======================== Entry Point ========================

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    warnings.filterwarnings("ignore")
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
