# -*- coding: utf-8 -*-
"""
AI_04_triconv.py (重写版)

此脚本用于训练全息图生成模型。
该版本已根据新的多通道数据集格式进行了重写：
-   输入 (X): 2通道 (目标振幅, 目标相位)
-   标签 (Y): 2通道 (重建振幅, 优化相位)
-   损失函数: L1(振幅) + MSE(相位) 的加权和
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

# 设置随机种子保证可重复性
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# <--- 新增: Matplotlib 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class ResidualBlock(nn.Module):
    """残差块 (与原版一致)"""
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
    """
    全息图生成模型 (架构修改)
    """
    # <--- 修改: input_channels 默认改为2
    def __init__(self, input_channels=2, output_channels=2, base_channels=64, num_res_blocks=3):
        super().__init__()
        # 初始编码层
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=7, padding=3), nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1), nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
        )
        # 残差块序列
        self.res_blocks = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_res_blocks)])
        
        # <--- 修改: 最终输出层, 输出通道改为2
        self.final_conv = nn.Conv2d(base_channels, output_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        x = self.final_conv(x)
        # <--- 修改: 移除了 Tanh 激活函数，直接输出原始值
        return x


def load_hdf5_data(hdf5_path):
    """
    修改: 加载HDF5数据以适应多通道格式
    """
    print(f"开始从 {hdf5_path} 加载多通道数据...")
    start_time = time.time()
    
    with h5py.File(hdf5_path, 'r') as f:
        sample_names = sorted([name for name in f.keys() if name.startswith('sample_')])
        num_samples = len(sample_names)
        
        print(f"找到 {num_samples} 个样本")
        
        first_sample = f[sample_names[0]]
        height, width = first_sample['input_amplitude'].shape
        print(f"数据形状: {height}x{width}")
        
        # <--- 修改: 预分配内存以适应双通道输入和输出
        X_data = np.zeros((num_samples, 2, height, width), dtype=np.float32)
        Y_data = np.zeros((num_samples, 2, height, width), dtype=np.float32)
        
        for i, sample_name in enumerate(tqdm(sample_names, desc="加载样本")):
            group = f[sample_name]
            # <--- 修改: 加载4个数据集并分别放入X和Y的通道中
            X_data[i, 0, :, :] = group['input_amplitude'][:]
            X_data[i, 1, :, :] = group['input_phase'][:]
            Y_data[i, 0, :, :] = group['output_amplitude'][:]
            Y_data[i, 1, :, :] = group['output_phase'][:]
    
    end_time = time.time()
    print(f"数据加载完成！耗时: {end_time - start_time:.2f} 秒")
    print(f"X_data shape: {X_data.shape}, Y_data shape: {Y_data.shape}")
    
    return X_data, Y_data


def create_data_loaders(X_data, Y_data, batch_size=16, train_ratio=0.85):
    """创建训练和验证数据加载器 (与原版一致)"""
    num_samples = X_data.shape[0]
    indices = np.random.permutation(num_samples)
    split_idx = int(num_samples * train_ratio)
    
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    
    X_train, Y_train = X_data[train_indices], Y_data[train_indices]
    X_val, Y_val = X_data[val_indices], Y_data[val_indices]
    
    print(f"训练集: {X_train.shape[0]} 个样本, 验证集: {X_val.shape[0]} 个样本")
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, device, num_epochs=100, learning_rate=1e-4, alpha=0.5):
    """
    修改: 训练模型，使用加权的混合损失函数
    """
    model.to(device)
    # <--- 新增: 定义两个损失函数
    criterion_mse = nn.MSELoss() # 用于相位
    criterion_l1 = nn.L1Loss()   # 用于振幅

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    history = {'train_loss': [], 'val_loss': [], 'train_amp_loss': [], 'train_phase_loss': [], 'val_amp_loss': [], 'val_phase_loss': []}
    best_val_loss = float('inf')
    
    print("开始训练...")
    print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Val Loss':<12} | {'Time':<8}")
    print("-" * 45)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        running_loss, amp_loss_t, phase_loss_t = 0.0, 0.0, 0.0
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            # <--- 修改: 分离通道并计算混合损失
            pred_amp, pred_phase = output[:, 0, :, :], output[:, 1, :, :]
            target_amp, target_phase = target[:, 0, :, :], target[:, 1, :, :]
            
            loss_a = criterion_l1(pred_amp, target_amp)
            loss_p = criterion_mse(pred_phase, target_phase)
            total_loss = loss_p + alpha * loss_a # 核心: 损失加权
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            amp_loss_t += loss_a.item()
            phase_loss_t += loss_p.item()
        
        history['train_loss'].append(running_loss / len(train_loader))
        history['train_amp_loss'].append(amp_loss_t / len(train_loader))
        history['train_phase_loss'].append(phase_loss_t / len(train_loader))
        
        # 验证阶段
        model.eval()
        val_loss, amp_loss_v, phase_loss_v = 0.0, 0.0, 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                # <--- 修改: 分离通道并计算混合损失
                pred_amp, pred_phase = output[:, 0, :, :], output[:, 1, :, :]
                target_amp, target_phase = target[:, 0, :, :], target[:, 1, :, :]
            
                loss_a = criterion_l1(pred_amp, target_amp)
                loss_p = criterion_mse(pred_phase, target_phase)
                total_loss = loss_p + alpha * loss_a
                
                val_loss += total_loss.item()
                amp_loss_v += loss_a.item()
                phase_loss_v += loss_p.item()
        
        epoch_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(epoch_val_loss)
        history['val_amp_loss'].append(amp_loss_v / len(val_loader))
        history['val_phase_loss'].append(phase_loss_v / len(val_loader))
        
        scheduler.step(epoch_val_loss)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_model_multichannel.pth')
        
        epoch_time = time.time() - epoch_start_time
        print(f"{epoch+1:>4}/{num_epochs} | {history['train_loss'][-1]:<12.6f} | {history['val_loss'][-1]:<12.6f} | {epoch_time:<8.2f}s")
    
    # <--- 新增: 绘制更详细的损失曲线
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='总训练损失 (Train Loss)')
    plt.plot(history['val_loss'], label='总验证损失 (Val Loss)')
    plt.title('总损失曲线 (Total Loss)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(history['train_amp_loss'], label='训练振幅损失 (L1)')
    plt.plot(history['val_amp_loss'], label='验证振幅损失 (L1)')
    plt.title('振幅损失 (Amplitude Loss)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(history['train_phase_loss'], label='训练相位损失 (MSE)')
    plt.plot(history['val_phase_loss'], label='验证相位损失 (MSE)')
    plt.title('相位损失 (Phase Loss)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_loss_multichannel.png')
    plt.close()
    
    return history


def visualize_results(model, val_loader, device, num_examples=5):
    """
    修改: 可视化预测结果，展示所有通道的对比
    """
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            if i >= num_examples:
                break
                
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 移动到CPU并转换为numpy
            data_np = data.cpu().numpy()[0]
            target_np = target.cpu().numpy()[0]
            output_np = output.cpu().numpy()[0]
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f'样本 {i+1} 预测结果对比', fontsize=16)

            # --- Row 1: 振幅对比 ---
            im_a1 = axes[0, 0].imshow(data_np[0], cmap='hot')
            axes[0, 0].set_title('输入 - 目标振幅')
            plt.colorbar(im_a1, ax=axes[0, 0])
            
            im_a2 = axes[0, 1].imshow(output_np[0], cmap='hot')
            axes[0, 1].set_title('预测 - 重建振幅')
            plt.colorbar(im_a2, ax=axes[0, 1])

            im_a3 = axes[0, 2].imshow(target_np[0], cmap='hot')
            axes[0, 2].set_title('标签 - 重建振幅')
            plt.colorbar(im_a3, ax=axes[0, 2])

            # --- Row 2: 相位对比 ---
            im_p1 = axes[1, 0].imshow(data_np[1], cmap='hsv')
            axes[1, 0].set_title('输入 - 目标相位')
            plt.colorbar(im_p1, ax=axes[1, 0])
            
            im_p2 = axes[1, 1].imshow(output_np[1], cmap='hsv')
            axes[1, 1].set_title('预测 - 优化相位')
            plt.colorbar(im_p2, ax=axes[1, 1])
            
            im_p3 = axes[1, 2].imshow(target_np[1], cmap='hsv')
            axes[1, 2].set_title('标签 - 优化相位')
            plt.colorbar(im_p3, ax=axes[1, 2])

            for ax_row in axes:
                for ax in ax_row:
                    ax.axis('off')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'result_comparison_multichannel_{i}.png')
            plt.close()


def main():
    """主函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 超参数设置
    BATCH_SIZE = 16
    NUM_EPOCHS = 200
    LEARNING_RATE = 2e-4
    TRAIN_RATIO = 0.9
    ALPHA = 0.5 # 振幅损失的权重
    
    # 1. 加载数据
    # <--- 修改: 指定新的数据集文件路径
    H5_PATH = '/home/rh/rh-apps/chatfront20250728/gs_traindata_multichan/gs_traindata_multichan_01/training_data_multichannel.h5'
    if not os.path.exists(H5_PATH):
        print(f"错误: 找不到数据集文件 {H5_PATH}")
        print("请先运行重写版的 GS_11_traindata_rh.py 来生成数据。")
        return
        
    X_data, Y_data = load_hdf5_data(H5_PATH)
    
    # 2. 创建数据加载器
    train_loader, val_loader = create_data_loaders(X_data, Y_data, BATCH_SIZE, TRAIN_RATIO)
    
    # 3. 初始化模型
    model = HologramGenerator()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 4. 训练模型
    train_model(
        model, train_loader, val_loader, device, NUM_EPOCHS, LEARNING_RATE, ALPHA
    )
    
    # 5. 加载最佳模型并进行可视化
    print("加载最佳模型进行可视化...")
    model.load_state_dict(torch.load('best_model_multichannel.pth'))
    visualize_results(model, val_loader, device, num_examples=5)
    
    print("\n训练完成！")
    print("最佳模型已保存为: best_model_multichannel.pth")
    print("训练曲线已保存为: training_loss_multichannel.png")
    print("结果对比图已保存为: result_comparison_multichannel_*.png")

if __name__ == "__main__":
    main()