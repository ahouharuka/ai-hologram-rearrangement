import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import time
import torch.nn as nn
import torch.nn.functional as F

def load_hdf5_to_numpy(hdf5_path, start_idx=0, end_idx=None):
    """
    将HDF5文件中的所有样本加载到numpy数组中
    """
    print(f"开始从 {hdf5_path} 加载数据...")
    start_time = time.time()
    
    with h5py.File(hdf5_path, 'r') as f:
        # 获取所有样本名并排序
        sample_names = sorted([name for name in f.keys() if name.startswith('sample_')])
        
        # 确定要加载的样本范围
        if end_idx is None:
            end_idx = len(sample_names)
        sample_names = sample_names[start_idx:end_idx]
        num_samples = len(sample_names)
        
        print(f"加载 {num_samples} 个样本")
        
        # 从第一个样本获取数据形状信息
        first_sample = f[sample_names[0]]
        height, width = first_sample['target_amplitude'].shape
        print(f"数据形状: {height}x{width}")
        
        # 预分配内存
        X_train = np.zeros((num_samples, 1, height, width), dtype=np.float32)
        Y_train = np.zeros((num_samples, 1, height, width), dtype=np.float32)
        
        # 逐个加载样本
        for i, sample_name in enumerate(sample_names):
            group = f[sample_name]
            
            # 加载输入数据 (target_amplitude)
            X_train[i, 0, :, :] = group['target_amplitude'][:]
            
            # 加载目标数据 (optimal_phase - 已经归一化)
            Y_train[i, 0, :, :] = group['optimal_phase'][:]
            
            if (i + 1) % 100 == 0:
                print(f"已加载 {i + 1}/{num_samples} 个样本")
    
    end_time = time.time()
    print(f"数据加载完成！耗时: {end_time - start_time:.2f} 秒")
    print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"Y_train shape: {Y_train.shape}, dtype: {Y_train.dtype}")
    
    return X_train, Y_train

def create_gpu_dataloader_corrected(X_train, Y_train, batch_size=32, train_ratio=0.8):
    """
    创建适用于GPU训练的数据加载器（修正版）
    """
    # 划分训练集和验证集
    num_samples = X_train.shape[0]
    split_idx = int(num_samples * train_ratio)
    
    X_train_split = X_train[:split_idx]
    Y_train_split = Y_train[:split_idx]
    X_val_split = X_train[split_idx:]
    Y_val_split = Y_train[split_idx:]
    
    print(f"训练集: {X_train_split.shape[0]} 个样本")
    print(f"验证集: {X_val_split.shape[0]} 个样本")
    
    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if device.type == 'cuda':
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 创建CPU上的数据集
    train_dataset = TensorDataset(
        torch.from_numpy(X_train_split), 
        torch.from_numpy(Y_train_split)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val_split), 
        torch.from_numpy(Y_val_split)
    )
    
    # 创建数据加载器 - 移除pin_memory或者只在CPU模式下使用
    pin_memory = (device.type == 'cpu')  # 只有在CPU上训练时才使用pin_memory
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # 对于小数据集，可以设置为0避免多进程问题
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, device



class ResidualBlock(nn.Module):
    """一个简单的残差块，包含两个卷积层和跳跃连接"""
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
        out += identity  # 跳跃连接
        out = self.relu(out)
        return out

class HologramGenerator(nn.Module):
    """生成全息图的AI模型"""
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
        # 最终输出层，预测归一化的相位值 [-1, 1] 对应 [-π, π]
        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=1, padding=0)
        # 使用Tanh激活将输出约束在[-1, 1]范围内
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x 的 shape: [batch_size, 1, height, width]
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        x = self.final_conv(x)
        x = self.tanh(x) 
        # 输出 shape: [batch_size, 1, height, width]，每个像素值在 [-1, 1]
        return x
    

def train_model_simple(model, train_loader, val_loader, device, num_epochs=100):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 移动数据到GPU
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss/len(val_loader):.6f}')


def create_simple_dataloader(X_train, Y_train, batch_size=32, train_ratio=0.8):
    """
    创建简单但有效的数据加载器
    """
    # 划分训练集和验证集
    num_samples = X_train.shape[0]
    split_idx = int(num_samples * train_ratio)
    
    # 创建数据集
    train_dataset = TensorDataset(
        torch.from_numpy(X_train[:split_idx]),
        torch.from_numpy(Y_train[:split_idx])
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_train[split_idx:]),
        torch.from_numpy(Y_train[split_idx:])
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader



# 主程序
if __name__ == "__main__":
    # 1. 加载数据到numpy数组
    X_train, Y_train = load_hdf5_to_numpy('gs_training_data/gs_training_data_01/training_data_normalized.h5')
    
    # 2. 创建简单数据加载器
    train_loader, val_loader = create_simple_dataloader(X_train, Y_train, batch_size=16)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 3. 初始化模型
    model = HologramGenerator().to(device)
    
    # 4. 开始训练
    train_model_simple(model, train_loader, val_loader, device, num_epochs=2)