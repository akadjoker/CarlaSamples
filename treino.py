import os
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
 
from model_light import LightSteeringNet

# ============================
# Configurações
# ============================
DATASET_DIR = 'dataset'
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-4
IMG_SIZE = (64, 64)
EARLY_STOPPING_PATIENCE = 5

# ============================
# Balanceamento
# ============================
def balance_dataset(image_paths, angles, threshold=0.05, keep_ratio=0.15):
    new_paths, new_angles = [], []
    for img, angle in zip(image_paths, angles):
        if abs(angle) < threshold:
            if random.random() < keep_ratio:
                new_paths.append(img)
                new_angles.append(angle)
        else:
            new_paths.append(img)
            new_angles.append(angle)
    return np.array(new_paths), np.array(new_angles)

# ============================
# Dataset
# ============================
class SteeringDataset(Dataset):
    def __init__(self, image_paths, steering_angles):
        self.image_paths = image_paths
        self.steering_angles = steering_angles
        self.transform = lambda img: torch.tensor(
            np.array(img.resize(IMG_SIZE)).transpose((2, 0, 1)) / 255.0,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        angle = self.steering_angles[idx]
        return image, torch.tensor(angle, dtype=torch.float32)


# ============================
# Carregar dados
# ============================
def load_sessions_data():
    image_paths, angles = [], []
    session_dirs = glob.glob(os.path.join(DATASET_DIR, 'session_*'))
    for session_dir in session_dirs:
        csv_path = os.path.join(session_dir, 'steering_data.csv')
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if 'image_path' not in df.columns or 'steering' not in df.columns:
            continue
        for _, row in df.iterrows():
            path = os.path.join(session_dir, row['image_path'])
            if os.path.exists(path):
                image_paths.append(path)
                angles.append(float(row['steering']))
    return np.array(image_paths), np.array(angles)

# ============================
# Treino
# ============================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_paths, angles = load_sessions_data()
    image_paths, angles = balance_dataset(image_paths, angles)
    dataset = SteeringDataset(image_paths, angles)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4)

    model = LightSteeringNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            preds = model(images)
            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                preds = model(images)
                loss = loss_fn(preds, targets)
                val_loss += loss.item()

        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        print(f"\nÉpoca {epoch+1}/{EPOCHS} - Treino Loss: {avg_train:.4f} - Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'model_light.pth')
            print("\nNovo melhor modelo salvo como 'model_light.pth'")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("\n🛑 Early stopping ativado")
                break

    # ============================
    # Gráfico
    # ============================
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Treino')
    plt.plot(val_losses, label='Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Loss (MSE)')
    plt.title('Treinamento - Modelo Light')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_light_plot.png')
    plt.show()

