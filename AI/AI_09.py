# -*- coding: utf-8 -*-
"""
AI_06_UNet_rh_weighted.py (U-Net + Weighted Complex Loss 版本)

改进点:
1. 架构升级为 U-Net，专门处理图像到图像的转换任务。
2. 引入 WeightedComplexMSELoss，给光斑区域加权，解决"全黑预测"问题。
3. 强制振幅输出为 [0,1]，防止背景数值漂移。
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

# 1. 加权物理损失函数 (核心修改)
class WeightedComplexMSELoss(nn.Module):
    """
    加权复数域均方误差损失。
    Loss = Mean( |Pred - Target|^2 * Weight_Map )
    
    机制:
    1. 计算复数距离误差。
    2. 根据 Target 的振幅构建权重图：光斑区域(Target>0.1)权重放大 N 倍。
    3. 强迫模型优先拟合光斑，绝不姑息"全黑"偷懒行为。
    """
    def __init__(self, weight_factor=100.0): # 默认给予100倍惩罚权重
        super().__init__()
        self.weight_factor = weight_factor

    def forward(self, pred, target):
        # pred/target shape: [B, 2, H, W]
        # Channel 0: Amplitude, Channel 1: Phase
        
        p_amp, p_phase = pred[:, 0], pred[:, 1]
        t_amp, t_phase = target[:, 0], target[:, 1]
        
        # Euler公式: A * e^(i*phi) = A * cos(phi) + i * A * sin(phi)
        p_real = p_amp * torch.cos(p_phase)
        p_imag = p_amp * torch.sin(p_phase)
        
        t_real = t_amp * torch.cos(t_phase)
        t_imag = t_amp * torch.sin(t_phase)
        
        # 计算每个像素的复数距离平方 (Pixel-wise Loss)
        # [B, H, W]
        loss_pixel = (p_real - t_real)**2 + (p_imag - t_imag)**2
        
        # === 关键步骤：创建空间权重矩阵 ===
        # 基础权重为 1
        spatial_weight = torch.ones_like(t_amp)
        # 在目标振幅 > 0.1 的地方(光斑区)，权重设为 weight_factor
        spatial_weight[t_amp > 0.1] = self.weight_factor 
        
        # 计算加权平均 Loss
        loss = torch.mean(loss_pixel * spatial_weight)
        
        return loss

# 2. U-Net 模型架构 (保持不变)
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
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1) # Skip connection
        x = self.dconv_up3(x)

        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1) # Skip connection
        x = self.dconv_up2(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1) # Skip connection
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        # 通道0 (振幅): 使用 Sigmoid 强制归一化到 [0, 1]
        amp = torch.sigmoid(out[:, 0:1, :, :]) 
        phase = out[:, 1:2, :, :]
        
        return torch.cat([amp, phase], dim=1)

def load_data(hdf5_path):
    print(f"Loading data from {hdf5_path}...")
    with h5py.File(hdf5_path, 'r') as f:
        keys = sorted(list(f.keys()))
        sample_count = len(keys) 
        print(f"Total samples found: {sample_count}")
        
        sample0 = f[keys[0]]
        h, w = sample0['input_amplitude'].shape
        
        X = np.zeros((sample_count, 2, h, w), dtype=np.float32)
        Y = np.zeros((sample_count, 2, h, w), dtype=np.float32)
        
        for i, k in enumerate(tqdm(keys)):
            g = f[k]
            X[i, 0] = g['input_amplitude'][:]
            X[i, 1] = g['input_phase'][:]
            Y[i, 0] = g['output_amplitude'][:]
            Y[i, 1] = g['output_phase'][:]
            
    return X, Y

def train(model, loader, val_loader, device, epochs=50): # 建议 Epoch 至少 50
    model.to(device)
    
    # === 修改: 使用加权 Loss，权重设为 100.0 ===
    criterion = WeightedComplexMSELoss(weight_factor=100.0)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # 耐心值 patience 可以稍微调大一点，因为前期 Loss 波动可能会变大
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.5)
    
    # 自动创建保存目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "results_unet02")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Start Training with Weighted Loss...")
    
    history = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in tqdm(loader, desc=f"Ep {epoch+1}", leave=False):
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
        history.append(avg_val)
        
        print(f"Ep {epoch+1} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")
        scheduler.step(avg_val)
        
        # 保存最佳模型
        if avg_val == min(history):
            torch.save(model.state_dict(), os.path.join(save_dir, "best_unet_model.pth"))

    # 训练结束后可视化
    infer_and_visualize(model, val_loader, device, save_dir)

def infer_and_visualize(model, val_loader, device, save_dir):
    model.eval()
    model.to(device)
    
    # 取一个样本
    x, y = next(iter(val_loader))
    x = x.to(device)
    
    with torch.no_grad():
        pred = model(x)
        pred_amp = pred[:, 0]
        pred_phase = pred[:, 1]
        
        # IFFT 重建全息图
        complex_field = pred_amp * torch.exp(1j * pred_phase)
        hologram_complex = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(complex_field)))
        hologram_phase = torch.angle(hologram_complex)
    
    # 绘图
    idx = 0
    in_amp = x[idx, 0].cpu().numpy()
    target_amp = y[idx, 0].cpu().numpy() # Ground Truth
    pred_a = pred_amp[idx].cpu().numpy()
    holo_p = hologram_phase[idx].cpu().numpy()
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. 输入
    axes[0].imshow(in_amp, cmap='hot', vmin=0, vmax=1)
    axes[0].set_title("Input (Ideal Traps)")
    
    # 2. 预测振幅 (强制vmin=0, vmax=1)
    # 加权 Loss 后，这里应该能看到亮斑了
    im2 = axes[1].imshow(pred_a, cmap='hot', vmin=0, vmax=1) 
    axes[1].set_title(f"UNet Pred Amp\nMean:{pred_a.mean():.3f}")
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    # 3. 标签振幅
    im3 = axes[2].imshow(target_amp, cmap='hot', vmin=0, vmax=1)
    axes[2].set_title("Target Amp (WGS)")
    plt.colorbar(im3, ax=axes[2], fraction=0.046)
    
    # 4. 最终全息图
    axes[3].imshow(holo_p, cmap='gray')
    axes[3].set_title("Final Hologram (Phase)")
    
    save_path = os.path.join(save_dir, "unet_result_weighted.png")
    plt.savefig(save_path)
    print(f"Result saved to: {save_path}")

def main():
    # 请确保路径正确
    H5_PATH = './ustc/Others/ai原子阵列重排/gs_training_data/gs_training_data_05/training_data_complex.h5'
    
    if not os.path.exists(H5_PATH):
        print(f"Error: Data file not found at {H5_PATH}")
        return

    X, Y = load_data(H5_PATH)
    
    # 转换为 Tensor
    split = int(0.9 * len(X))
    train_ds = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(Y[:split]))
    val_ds = TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(Y[split:]))
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)
    
    model = UNet()
    train(model, train_loader, val_loader, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

if __name__ == "__main__":
    main()