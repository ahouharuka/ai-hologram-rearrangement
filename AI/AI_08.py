# -*- coding: utf-8 -*-
"""
AI_06_UNet_rh.py (U-Net + Complex Loss 版本)

改进点:
1. 架构升级为 U-Net，专门处理图像到图像的转换任务。
2. 引入 ComplexMSELoss，完美解决相位在暗处无法定义的问题。
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

# 1. 物理损失函数 (核心修改)
class ComplexMSELoss(nn.Module):
    """
    复数域均方误差损失。
    Loss = |Pred_Complex - Target_Complex|^2
    这会自动忽略振幅为0区域的相位误差，这是物理上最合理的Loss。
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred/target shape: [B, 2, H, W]
        # Channel 0: Amplitude, Channel 1: Phase
        
        # 构建复数场
        p_amp, p_phase = pred[:, 0], pred[:, 1]
        t_amp, t_phase = target[:, 0], target[:, 1]
        
        # Euler公式: A * e^(i*phi) = A * cos(phi) + i * A * sin(phi)
        p_real = p_amp * torch.cos(p_phase)
        p_imag = p_amp * torch.sin(p_phase)
        
        t_real = t_amp * torch.cos(t_phase)
        t_imag = t_amp * torch.sin(t_phase)
        
        # 计算复数距离的平方: |z1 - z2|^2 = (real1-real2)^2 + (imag1-imag2)^2
        loss = (p_real - t_real)**2 + (p_imag - t_imag)**2
        
        return torch.mean(loss)

# 2. U-Net 模型架构 (核心修改)
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
        
        # === 关键修正 ===
        # 通道0 (振幅): 使用 Sigmoid 强制归一化到 [0, 1]
        # 通道1 (相位): 不做处理 (或者也可以用 HardTanh 限制范围，但在复数loss下没必要)
        amp = torch.sigmoid(out[:, 0:1, :, :]) 
        phase = out[:, 1:2, :, :]
        
        return torch.cat([amp, phase], dim=1)

def load_data(hdf5_path):
    print(f"Loading data from {hdf5_path}...")
    with h5py.File(hdf5_path, 'r') as f:
        keys = sorted(list(f.keys()))
        # 简单取前200个样本做快速验证，如果跑通了再由用户加大
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

def train(model, loader, val_loader, device, epochs=30):
    model.to(device)
    criterion = ComplexMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # 自动创建保存目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "results_unet01")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Start Training...")
    
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
        # 注意: 如果训练数据的Label也是空域，那么重建全息图需要IFFT
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
    
    # 2. 预测振幅 (强制vmin=0, vmax=1 以检查背景是否黑)
    im2 = axes[1].imshow(pred_a, cmap='hot', vmin=0, vmax=1) 
    axes[1].set_title(f"UNet Pred Amp\nMean:{pred_a.mean():.3f}")
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    # 3. 标签振幅 (对比用)
    im3 = axes[2].imshow(target_amp, cmap='hot', vmin=0, vmax=1)
    axes[2].set_title("Target Amp (WGS)")
    plt.colorbar(im3, ax=axes[2], fraction=0.046)
    
    # 4. 最终全息图
    axes[3].imshow(holo_p, cmap='gray')
    axes[3].set_title("Final Hologram (Phase)")
    
    save_path = os.path.join(save_dir, "unet_result.png")
    plt.savefig(save_path)
    print(f"Result saved to: {save_path}")

def main():
    # 请确保路径正确指向你的 h5 文件
    H5_PATH = './ustc/Others/ai原子阵列重排/gs_training_data/gs_training_data_03/training_data_multichannel_fixed.h5'
    
    # 本地测试路径 (如果需要)
    # H5_PATH = r"你的本地路径/training_data_multichannel_fixed.h5"

    if not os.path.exists(H5_PATH):
        print(f"Error: Data file not found at {H5_PATH}")
        return

    X, Y = load_data(H5_PATH)
    
    # 转换为 Tensor
    split = int(0.9 * len(X))
    train_ds = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(Y[:split]))
    val_ds = TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(Y[split:]))
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True) # Batch size 稍微调小一点适应显存
    val_loader = DataLoader(val_ds, batch_size=8)
    
    model = UNet()
    train(model, train_loader, val_loader, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

if __name__ == "__main__":
    main()