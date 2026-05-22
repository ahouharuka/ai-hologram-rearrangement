import h5py

def quick_view_h5(file_path):
    """
    快速查看HDF5文件内容
    """
    with h5py.File(file_path, 'r') as f:
        print("文件结构:")
        print("-" * 30)
        
        # 显示属性
        print("属性:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        
        print("\n数据集:")
        # 显示所有样本组
        for group_name in f.keys():
            print(f"  {group_name}:")
            group = f[group_name]
            for dataset_name in group.keys():
                dataset = group[dataset_name]
                print(f"    {dataset_name}: shape={dataset.shape}, dtype={dataset.dtype}")

# 使用示例
quick_view_h5('gs_training_data/gs_training_data_01/training_data_normalized.h5')