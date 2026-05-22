import h5py
import numpy as np
from tqdm import tqdm  # 用于显示进度条

def normalize_all_samples(input_path, output_path):
    """
    处理所有样本组中的optimal_phase数据
    """
    
    with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        
        print("开始处理所有样本...")
        
        # 获取所有样本组名并排序
        sample_names = sorted([name for name in f_in.keys() if name.startswith('sample_')])
        print(f"找到 {len(sample_names)} 个样本")
        
        # 用于统计
        total_samples = len(sample_names)
        processed_samples = 0
        
        # 使用进度条
        for sample_name in tqdm(sample_names, desc="处理样本"):
            group_in = f_in[sample_name]
            group_out = f_out.create_group(sample_name)
            
            # 检查该样本是否包含optimal_phase
            if 'optimal_phase' not in group_in:
                print(f"警告: {sample_name} 中未找到 optimal_phase，跳过")
                continue
            
            # 复制所有其他数据集
            for dataset_name in group_in.keys():
                if dataset_name != 'optimal_phase':
                    try:
                        data = group_in[dataset_name][:]
                        group_out.create_dataset(dataset_name, data=data)
                    except Exception as e:
                        print(f"  复制 {dataset_name} 时出错: {e}")
            
            # 处理optimal_phase数据集
            try:
                original_phase = group_in['optimal_phase'][:]
                
                # 记录原始范围（用于调试）
                original_min = original_phase.min()
                original_max = original_phase.max()
                
                # 归一化：[-π, π] -> [-1, 1]
                normalized_phase = original_phase / np.pi
                
                # 创建归一化后的数据集
                dset_out = group_out.create_dataset('optimal_phase', data=normalized_phase)
                
                # 复制原始属性
                original_dset = group_in['optimal_phase']
                for attr_name, attr_value in original_dset.attrs.items():
                    dset_out.attrs[attr_name] = attr_value
                
                # 添加归一化信息
                dset_out.attrs['normalization'] = 'divided_by_pi'
                dset_out.attrs['original_range'] = f"[{original_min:.6f}, {original_max:.6f}]"
                dset_out.attrs['normalized_range'] = f"[{normalized_phase.min():.6f}, {normalized_phase.max():.6f}]"
                
                processed_samples += 1
                
            except Exception as e:
                print(f"处理 {sample_name}/optimal_phase 时出错: {e}")
        
        print(f"处理完成！成功处理 {processed_samples}/{total_samples} 个样本")

# 安装tqdm（如果还没有的话）：pip install tqdm
# 执行归一化
normalize_all_samples('gs_training_data_rh/gs_training_data_rh_03/training_data.h5', 'gs_training_data_rh/gs_training_data_rh_03/training_data_normalized.h5')