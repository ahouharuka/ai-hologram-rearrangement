# -*- coding: utf-8 -*-
"""
AI_05_multichan_rh_fixed.py (修正版)

修正重点:
1. 损失函数: 使用 Phase-Aware Loss 处理周期性问题。
2. 推理流程: CNN 输出空域信息 -> IFFT -> 得到 Hologram。
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 设置随机种子
torch.manual_seed(42)

# 可视化设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class HologramGenerator(nn.Module):
    def __init__(self, input_channels=2, output_channels=2, base_channels=64, num_res_blocks=6): # 增加深度
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_res_blocks)])
        self.final_conv = nn.Conv2d(base_channels, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        x = self.final_conv(x)
        return x

def load_data(hdf5_path):
    print(f"Loading from {hdf5_path}...")
    with h5py.File(hdf5_path, 'r') as f:
        keys = sorted(list(f.keys()))
        sample0 = f[keys[0]]
        h, w = sample0['input_amplitude'].shape
        n = len(keys)
        
        X = np.zeros((n, 2, h, w), dtype=np.float32)
        Y = np.zeros((n, 2, h, w), dtype=np.float32)
        
        for i, k in enumerate(tqdm(keys)):
            g = f[k]
            X[i, 0] = g['input_amplitude'][:]
            X[i, 1] = g['input_phase'][:]
            Y[i, 0] = g['output_amplitude'][:]
            Y[i, 1] = g['output_phase'][:] # 这是空域相位
            
    return X, Y

# === 关键修正: 损失函数 ===
class PhaseAwareLoss(nn.Module):
    """
    处理相位的周期性。
    L2 Loss on (e^i*pred - e^i*target)
    等价于 2 * (1 - cos(pred - target))
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        pred_amp, pred_phase = pred[:, 0], pred[:, 1]
        target_amp, target_phase = target[:, 0], target[:, 1]
        
        # 1. 振幅损失 (L1)
        loss_amp = self.l1(pred_amp, target_amp)
        
        # 2. 相位损失 (Cosine Distance)
        # 避免直接用 MSE 导致 -pi 和 pi 误差巨大
        diff = pred_phase - target_phase
        loss_phase = torch.mean(1 - torch.cos(diff))
        
        return loss_phase + self.alpha * loss_amp, loss_amp, loss_phase

def train(model, loader, val_loader, device, epochs=50):
    model.to(device)
    criterion = PhaseAwareLoss(alpha=1.0) # 振幅和相位权重
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in tqdm(loader, desc=f"Epoch {epoch+1}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss, l_a, l_p = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss, _, _ = criterion(out, y)
                val_loss += loss.item()
        
        avg_val = val_loss/len(val_loader)
        print(f"Ep {epoch+1} | Train: {total_loss/len(loader):.4f} | Val: {avg_val:.4f}")
        scheduler.step(avg_val)
        
    torch.save(model.state_dict(), "best_model_fixed.pth")

def infer_and_visualize(model, val_loader, device):
    """
    修正后的可视化：包含 IFFT 步骤以生成全息图
    """
    model.eval()
    model.to(device)
    
    x, y = next(iter(val_loader))
    x = x.to(device)
    
    with torch.no_grad():
        # 1. CNN 预测空域复振幅
        pred = model(x) # [B, 2, H, W]
        pred_amp = pred[:, 0]
        pred_phase = pred[:, 1]
        
        # 2. 关键步骤: 组合复数场并执行 IFFT
        # Paper: "apply inverse FFT to transform predictions into final hologram"
        # 注意 PyTorch 的维度: [B, H, W]
        complex_field_spatial = pred_amp * torch.exp(1j * pred_phase)
        
        # 假设光学系统是从 SLM -> Lens -> Atom 是 FFT
        # 那么从 Atom (Spatial) -> Lens -> SLM (Hologram) 应该是 IFFT
        # 注意 fftshift 的处理需要与训练数据生成时保持一致（反向）
        hologram_complex = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(complex_field_spatial)))
        
        # 3. 提取最终全息图相位
        hologram_phase = torch.angle(hologram_complex)
        
    # 绘图
    idx = 0
    in_amp = x[idx, 0].cpu().numpy()
    pred_a = pred_amp[idx].cpu().numpy()
    pred_p = pred_phase[idx].cpu().numpy()
    holo_p = hologram_phase[idx].cpu().numpy()
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(in_amp, cmap='hot')
    axes[0].set_title("Input: Target Traps")
    
    axes[1].imshow(pred_a, cmap='hot')
    axes[1].set_title("CNN Output: Focal Amp")
    
    axes[2].imshow(pred_p, cmap='hsv')
    axes[2].set_title("CNN Output: Focal Phase")
    
    axes[3].imshow(holo_p, cmap='gray')
    axes[3].set_title("Final Hologram (SLM Phase)") # 这就是你要加载到 SLM 上的图
    
    plt.savefig("inference_result_fixed.png")
    print("Result saved.")

def main():
    H5_PATH = './ustc/Others/ai原子阵列重排/gs_training_data/gs_training_data_test_03/training_data_multichannel_fixed.h5'
    if not os.path.exists(H5_PATH):
        print("Run GS_13 first!")
        return

    X, Y = load_data(H5_PATH)
    
    # 简单划分
    split = int(0.9 * len(X))
    train_ds = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(Y[:split]))
    val_ds = TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(Y[split:]))
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    
    model = HologramGenerator()
    train(model, train_loader, val_loader, torch.device('cuda'))
    
    infer_and_visualize(model, val_loader, torch.device('cuda'))

if __name__ == "__main__":
    main()