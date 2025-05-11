import os
import random
import glob
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

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

import timm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torchmetrics.classification import MulticlassF1Score

# ======================== Setup & Cleanup ========================

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# ======================== Configuration ========================

CFG = {
    'IMG_SIZE': 448,
    'EPOCHS': 15,
    'LEARNING_RATE': 1e-4,
    'BATCH_SIZE': 8,
    'SEED': 42,
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

class LitModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, batch):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        return loss, outputs, labels

# ======================== Train & Validation ========================

def validate(model, val_loader, rank):
    model.eval()
    model.module.f1.reset()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"[GPU {rank}] validation"):
            images = images.to(rank, non_blocking=True)
            labels = labels.to(rank, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            model.module.f1.update(preds, labels)

    avg_loss = total_loss / len(val_loader)
    f1_score = model.module.f1.compute().item()

    return avg_loss, f1_score

# ======================== Training ========================

def train(rank, world_size):
    setup(rank, world_size)
    seed_everything(CFG['SEED'])

    # === 데이터 준비 ===
    all_img_list = glob.glob('../data/train/*/*')
    df = pd.DataFrame({'img_path': all_img_list})
    df['rock_type'] = df['img_path'].apply(lambda x: x.split('/')[3])
    le = preprocessing.LabelEncoder()
    df['rock_type'] = le.fit_transform(df['rock_type'])
    df = df.sample(5000)
    train_df, val_df = train_test_split(df, test_size=0.3, stratify=df['rock_type'], random_state=CFG['SEED'])

    # === transform ===
    model_tmp = timm.create_model(CFG['MODEL_NAME'], pretrained=True)
    config = timm.data.resolve_data_config({}, model=model_tmp)
    train_transform = A.Compose([
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.Normalize(mean=config['mean'], std=config['std']),
        ToTensorV2()
    ])

    # === dataset & dataloader ===
    train_dataset = CustomDataset(train_df['img_path'].values, train_df['rock_type'].values, transform=train_transform)
    val_dataset = CustomDataset(val_df['img_path'].values, val_df['rock_type'].values, transform=train_transform)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], sampler=val_sampler, num_workers=4, pin_memory=True)

    # === 모델, 옵티마이저, DDP 래핑 ===
    model = LitModel(CFG['MODEL_NAME'], CFG['NUM_CLASSES']).to(rank)
    model = DDP(model, device_ids=[rank])
    optimizer = optim.AdamW(model.parameters(), lr=CFG['LEARNING_RATE'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['EPOCHS'])

    best_f1 = 0.0

    for epoch in range(CFG['EPOCHS']):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"[GPU {rank}] Epoch {epoch+1}/{CFG['EPOCHS']}"):
            images, labels = batch
            images = images.to(rank, non_blocking=True)
            labels = labels.to(rank, non_blocking=True)

            optimizer.zero_grad()
            loss, _, _ = model.module.compute_loss((images, labels))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        val_loss, val_f1 = validate(model, val_loader, rank)

        if rank == 0:
            print(f"[Epoch {epoch+1}] Train Loss: {total_loss/len(train_loader):.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
            
            torch.save(model.module.state_dict(), f"./{epoch+1}_{CFG['MODEL_NAME']}.pth")

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.module.state_dict(), CFG['SAVE_PATH'])
                print(f"✅ [Epoch {epoch+1}] Best model saved (F1: {best_f1:.4f})")

    cleanup()

# ======================== Entry Point ========================

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
