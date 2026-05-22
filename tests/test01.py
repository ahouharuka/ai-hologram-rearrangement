def verify_normalization_results(file_path, num_samples_to_check=5):
    """验证归一化结果"""
    with h5py.File(file_path, 'r') as f:
        print("验证归一化结果:")
        
        # 获取所有样本名
        sample_names = sorted([name for name in f.keys() if name.startswith('sample_')])
        
        # 检查前几个样本
        for i, sample_name in enumerate(sample_names[:num_samples_to_check]):
            if 'optimal_phase' in f[sample_name]:
                phase_data = f[sample_name]['optimal_phase'][:]
                attrs = dict(f[sample_name]['optimal_phase'].attrs)
                
                print(f"\n{sample_name}/optimal_phase:")
                print(f"  数据范围: [{phase_data.min():.6f}, {phase_data.max():.6f}]")
                print(f"  数据形状: {phase_data.shape}")
                if 'normalization' in attrs:
                    print(f"  归一化信息: {attrs['normalization']}")
                if 'original_range' in attrs:
                    print(f"  原始范围: {attrs['original_range']}")

# 验证结果
verify_normalization_results('training_data_normalized.h5', num_samples_to_check=3)