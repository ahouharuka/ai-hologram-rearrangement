import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import time

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

def create_gpu_dataloader(X_train, Y_train, batch_size=32, train_ratio=0.8):
    """
    创建适用于GPU训练的数据加载器
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
    
    # 转换为PyTorch张量并移动到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 直接在GPU上创建张量，避免CPU到GPU的数据传输瓶颈
    X_train_tensor = torch.from_numpy(X_train_split).to(device)
    Y_train_tensor = torch.from_numpy(Y_train_split).to(device)
    X_val_tensor = torch.from_numpy(X_val_split).to(device)
    Y_val_tensor = torch.from_numpy(Y_val_split).to(device)
    
    # 创建数据集和数据加载器
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    
    # 使用pin_memory=True加速CPU到GPU的数据传输（如果数据在CPU上）
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, device


# 主程序
if __name__ == "__main__":
    # 1. 加载数据到numpy数组
    X_train, Y_train = load_hdf5_to_numpy('gs_training_data/gs_training_data_01/training_data_normalized.h5')
    
    # 2. 创建GPU数据加载器
    train_loader, val_loader, device = create_gpu_dataloader(X_train, Y_train, batch_size=16)
    
    # 3. 检查数据
    print("\n数据检查:")
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Input shape: {data.shape}, device: {data.device}")
        print(f"  Target shape: {target.shape}, device: {target.device}")
        print(f"  Input range: [{data.min():.3f}, {data.max():.3f}]")
        print(f"  Target range: [{target.min():.3f}, {target.max():.3f}]")
        
        if batch_idx >= 2:  # 只检查前几个batch
            break