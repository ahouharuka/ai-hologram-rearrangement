import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

# 设置随机种子保证可重复性
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class HologramGenerator(nn.Module):
    """全息图生成模型"""
    def __init__(self, input_channels=1, base_channels=64, num_res_blocks=3):
        super().__init__()
        # 初始编码层
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        # 残差块序列
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_res_blocks)]
        )
        # 最终输出层
        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=1, padding=0)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        x = self.final_conv(x)
        x = self.tanh(x)
        return x

def load_hdf5_data(hdf5_path):
    """加载HDF5数据到numpy数组"""
    print(f"开始从 {hdf5_path} 加载数据...")
    start_time = time.time()
    
    with h5py.File(hdf5_path, 'r') as f:
        # 获取所有样本名并排序
        sample_names = sorted([name for name in f.keys() if name.startswith('sample_')])
        num_samples = len(sample_names)
        
        print(f"找到 {num_samples} 个样本")
        
        # 从第一个样本获取数据形状
        first_sample = f[sample_names[0]]
        height, width = first_sample['target_amplitude'].shape
        print(f"数据形状: {height}x{width}")
        
        # 预分配内存
        X_data = np.zeros((num_samples, 1, height, width), dtype=np.float32)
        Y_data = np.zeros((num_samples, 1, height, width), dtype=np.float32)
        
        # 加载所有数据
        for i, sample_name in enumerate(tqdm(sample_names, desc="加载样本")):
            group = f[sample_name]
            X_data[i, 0, :, :] = group['target_amplitude'][:]
            Y_data[i, 0, :, :] = group['optimal_phase'][:]
    
    end_time = time.time()
    print(f"数据加载完成！耗时: {end_time - start_time:.2f} 秒")
    print(f"X_data shape: {X_data.shape}, range: [{X_data.min():.3f}, {X_data.max():.3f}]")
    print(f"Y_data shape: {Y_data.shape}, range: [{Y_data.min():.3f}, {Y_data.max():.3f}]")
    
    return X_data, Y_data

def create_data_loaders(X_data, Y_data, batch_size=16, train_ratio=0.8):
    """创建训练和验证数据加载器"""
    # 划分训练集和验证集
    num_samples = X_data.shape[0]
    split_idx = int(num_samples * train_ratio)
    
    indices = np.random.permutation(num_samples)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    X_train, Y_train = X_data[train_indices], Y_data[train_indices]
    X_val, Y_val = X_data[val_indices], Y_data[val_indices]
    
    print(f"训练集: {X_train.shape[0]} 个样本")
    print(f"验证集: {X_val.shape[0]} 个样本")
    
    # 创建数据集
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(Y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(Y_val)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, device, num_epochs=100, learning_rate=1e-4):
    """训练模型"""
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("开始训练...")
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'Time':<8} | {'LR':<8}")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train", leave=False):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 学习率调整
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"{epoch+1:>4}/{num_epochs} | {train_loss:>10.6f} | {val_loss:>10.6f} | {epoch_time:>6.1f}s | {current_lr:>8.2e}")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.close()
    
    return train_losses, val_losses

def evaluate_model(model, val_loader, device):
    """评估模型性能"""
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
    
    avg_loss = total_loss / len(val_loader)
    print(f"验证集平均损失: {avg_loss:.6f}")
    return avg_loss

def visualize_results(model, val_loader, device, num_examples=3):
    """可视化一些预测结果"""
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            if i >= num_examples:
                break
                
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 转换为numpy用于可视化
            data_np = data.cpu().numpy()[0, 0]
            target_np = target.cpu().numpy()[0, 0]
            output_np = output.cpu().numpy()[0, 0]
            
            # 绘制对比图
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(data_np, cmap='viridis')
            axes[0].set_title('Input Amplitude')
            axes[0].axis('off')
            
            axes[1].imshow(target_np, cmap='viridis')
            axes[1].set_title('Target Phase')
            axes[1].axis('off')
            
            axes[2].imshow(output_np, cmap='viridis')
            axes[2].set_title('Predicted Phase')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'result_comparison_{i}.png')
            plt.close()

def main():
    """主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"可用内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 超参数设置
    batch_size = 16
    num_epochs = 200
    learning_rate = 1e-4
    train_ratio = 0.8
    
    # 1. 加载数据
    X_data, Y_data = load_hdf5_data('/home/rh/rh-apps/chatfront20250728/gs_training_data_03/training_data_normalized.h5')
    
    # 2. 创建数据加载器
    train_loader, val_loader = create_data_loaders(X_data, Y_data, batch_size, train_ratio)
    
    # 3. 初始化模型
    model = HologramGenerator()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. 训练模型
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, device, num_epochs, learning_rate
    )
    
    # 5. 加载最佳模型并评估
    model.load_state_dict(torch.load('best_model.pth'))
    final_loss = evaluate_model(model, val_loader, device)

    if input('do u need visualization? 0/1: '):

    
    # 6. 可视化一些结果
        visualize_results(model, val_loader, device)
    
    print("训练完成！")
    print(f"最终验证损失: {final_loss:.6f}")
    print("最佳模型已保存为: best_model.pth")
    print("训练曲线已保存为: training_loss.png")
    print("结果对比图已保存为: result_comparison_*.png")

if __name__ == "__main__":
    main()