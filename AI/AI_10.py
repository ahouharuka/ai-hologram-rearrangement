# -*- coding: utf-8 -*-
"""
AI_07_Paper_Repro_Train.py (PRL 论文复刻训练版)

逻辑对应:
1. Input: 坐标生成的理想振幅 + 【从无约束WGS提取的自然相位】
2. Label: 物理上真实的复振幅场 (经过孔径截断模拟)
3. 目的: 学习 f(Coord, Teacher_Phase) -> Physical_Field

数据来源: GS_20_Paper_LowRes.py
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 设置随机种子，保证复现性
torch.manual_seed(42)

# Matplotlib 中文设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 加权复数损失函数 (保持高压)
# ==========================================
class WeightedComplexMSELoss(nn.Module):
    def __init__(self, weight_factor=200.0): # 建议提高到 200，强迫相位对齐
        super().__init__()
        self.weight_factor = weight_factor

    def forward(self, pred, target):
        # Channel 0: Amp, Channel 1: Phase
        p_amp, p_phase = pred[:, 0], pred[:, 1]
        t_amp, t_phase = target[:, 0], target[:, 1]
        
        # 转复数域
        p_real = p_amp * torch.cos(p_phase)
        p_imag = p_amp * torch.sin(p_phase)
        t_real = t_amp * torch.cos(t_phase)
        t_imag = t_amp * torch.sin(t_phase)
        
        loss_pixel = (p_real - t_real)**2 + (p_imag - t_imag)**2
        
        # 空间加权: 只有光镊位置(Target > 0.1)才重要
        spatial_weight = torch.ones_like(t_amp)
        spatial_weight[t_amp > 0.1] = self.weight_factor 
        
        return torch.mean(loss_pixel * spatial_weight)

# ==========================================
# 2. U-Net 模型 (标准版)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()
        self.dconv_down1 = DoubleConv(in_channels, 32)
        self.dconv_down2 = DoubleConv(32, 64)
        self.dconv_down3 = DoubleConv(64, 128)
        self.dconv_down4 = DoubleConv(128, 256)        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = DoubleConv(128 + 256, 128)
        self.dconv_up2 = DoubleConv(64 + 128, 64)
        self.dconv_up1 = DoubleConv(32 + 64, 32)
        self.conv_last = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x1 = self.maxpool(conv1)
        conv2 = self.dconv_down2(x1)
        x2 = self.maxpool(conv2)
        conv3 = self.dconv_down3(x2)
        x3 = self.maxpool(conv3)
        x4 = self.dconv_down4(x3)
        
        x = self.upsample(x4)        
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        # 振幅归一化 [0,1]，相位自由
        amp = torch.sigmoid(out[:, 0:1, :, :]) 
        phase = out[:, 1:2, :, :]
        return torch.cat([amp, phase], dim=1)

# ==========================================
# 3. 辅助功能
# ==========================================
def plot_loss_curves(history, save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train'], 'b-', label='Train Loss')
    plt.plot(history['val'], 'r-', label='Val Loss')
    plt.title('Loss Curve (Paper Reproduction)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log') # 对数坐标看细节
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

def load_data(hdf5_path):
    print(f"Loading data from {hdf5_path}...")
    with h5py.File(hdf5_path, 'r') as f:
        keys = sorted(list(f.keys()))
        sample_count = len(keys)
        print(f"Total samples: {sample_count}")
        
        # 读取第一张图获取尺寸
        h, w = f[keys[0]]['input_amplitude'].shape
        X = np.zeros((sample_count, 2, h, w), dtype=np.float32)
        Y = np.zeros((sample_count, 2, h, w), dtype=np.float32)
        
        for i, k in enumerate(tqdm(keys, desc="Loading")):
            g = f[k]
            # X: Channel 0 = Amp, Channel 1 = Extracted Phase (Teacher)
            X[i, 0] = g['input_amplitude'][:]
            X[i, 1] = g['input_phase'][:]
            # Y: Physical Field Label
            Y[i, 0] = g['output_amplitude'][:]
            Y[i, 1] = g['output_phase'][:]
            
    return X, Y

def infer_and_visualize(model, val_loader, device, save_dir):
    model.eval()
    model.to(device)
    x, y = next(iter(val_loader))
    x = x.to(device)
    
    with torch.no_grad():
        pred = model(x)
    
    # 取第一个样本可视化
    idx = 0
    # CPU numpy conversion
    in_amp = x[idx, 0].cpu().numpy()
    in_phase = x[idx, 1].cpu().numpy() # 这是 Teacher Phase
    
    target_amp = y[idx, 0].cpu().numpy()
    target_phase = y[idx, 1].cpu().numpy()
    
    pred_amp = pred[idx, 0].cpu().numpy()
    pred_phase = pred[idx, 1].cpu().numpy() # 这是 AI 学到的 Phase
    
    # IFFT 重建全息图 (验证用)
    complex_field = torch.tensor(pred_amp) * torch.exp(1j * torch.tensor(pred_phase))
    holo_complex = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(complex_field)))
    holo_phase = torch.angle(holo_complex).numpy()
    
    # === 增强版绘图: 重点对比 Input Phase 和 Pred Phase ===
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Paper Reproduction Verification", fontsize=16)
    
    # 第一行: 振幅对比
    axes[0,0].imshow(in_amp, cmap='hot')
    axes[0,0].set_title("Input Amp (Coords)")
    
    im1 = axes[0,1].imshow(pred_amp, cmap='hot', vmin=0, vmax=1)
    axes[0,1].set_title(f"AI Pred Amp\nMean:{pred_amp.mean():.4f}")
    plt.colorbar(im1, ax=axes[0,1])
    
    axes[0,2].imshow(target_amp, cmap='hot', vmin=0, vmax=1)
    axes[0,2].set_title("Label Amp (Physical)")
    
    axes[0,3].imshow(holo_phase, cmap='gray')
    axes[0,3].set_title("Generated Hologram")
    
    # 第二行: 相位对比 (关键!)
    # 我们只关心光斑位置的相位，背景相位是混乱的，用 Mask 过滤一下显示
    mask = in_amp > 0.1
    
    def mask_phase(phi, m):
        p = phi.copy()
        p[~m] = 0 # 背景置0，方便观察
        return p

    axes[1,0].imshow(mask_phase(in_phase, mask), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[1,0].set_title("Input Phase (Teacher)\nExtracted from WGS")
    
    axes[1,1].imshow(mask_phase(pred_phase, mask), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[1,1].set_title("AI Pred Phase (Student)\nShould Match Teacher")
    
    axes[1,2].imshow(mask_phase(target_phase, mask), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[1,2].set_title("Label Phase (Physical)")
    
    axes[1,3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "paper_repro_result.png"))
    print(f"Result saved to {os.path.join(save_dir, 'paper_repro_result.png')}")

# ==========================================
# 4. 主训练循环
# ==========================================
def train(model, loader, val_loader, device, epochs=50):
    model.to(device)
    # 权重设高一点，因为相位学习很难
    criterion = WeightedComplexMSELoss(weight_factor=200.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "results_paper_repro")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Start Training (Paper Reproduction Mode)...")
    history = {'train': [], 'val': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in tqdm(loader, desc=f"Ep {epoch+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
        
        avg_train = train_loss / len(loader)
        avg_val = val_loss / len(val_loader)
        history['train'].append(avg_train)
        history['val'].append(avg_val)
        
        print(f"Ep {epoch+1} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")
        scheduler.step(avg_val)
        
        if avg_val == min(history['val']):
            torch.save(model.state_dict(), os.path.join(save_dir, "best_unet_model.pth"))

    plot_loss_curves(history, save_dir)
    infer_and_visualize(model, val_loader, device, save_dir)

def main():
    # ⚠️ 确保指向 GS_20 生成的数据集
    H5_PATH = './ustc/Others/ai原子阵列重排/gs_training_data/gs_training_data_paper_01/training_data_paper_repro.h5'
    
    if not os.path.exists(H5_PATH):
        print(f"Error: Data file not found at {H5_PATH}")
        return

    X, Y = load_data(H5_PATH)
    
    # 简单的 Train/Val 分割
    split = int(0.9 * len(X))
    train_ds = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(Y[:split]))
    val_ds = TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(Y[split:]))
    
    # 256x256 比较小，Batch Size 可以开大一点，比如 32
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    
    model = UNet()
    train(model, train_loader, val_loader, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

if __name__ == "__main__":
    main()