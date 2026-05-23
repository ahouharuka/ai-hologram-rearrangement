import numpy as np
import cupy as cp  # 导入CuPy
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import ndimage
from matplotlib import font_manager
import time
import h5py
import os

# 设置使用GPU设备
cp.cuda.Device(0).use()  # 使用第一个GPU

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class WeightedGSAlgorithm:
    def __init__(self, target_amplitude, weights=None, slm_size=(512, 512), 
                 iterations=50, phase_initial=None, use_gpu=True):
        """
        初始化加权GS算法
        
        参数:
        target_amplitude: 目标振幅分布
        weights: 权重矩阵
        slm_size: SLM的尺寸
        iterations: 迭代次数
        phase_initial: 初始相位估计
        use_gpu: 是否使用GPU加速
        """
        self.use_gpu = use_gpu
        self.slm_size = slm_size
        self.iterations = iterations
        
        # 将数据转移到GPU
        if use_gpu:
            self.target_amplitude = cp.asarray(target_amplitude)
            if weights is None:
                self.weights = cp.ones_like(self.target_amplitude)
            else:
                self.weights = cp.asarray(weights)
            
            if phase_initial is None:
                self.phase = cp.random.rand(*slm_size) * 2 * cp.pi
            else:
                self.phase = cp.asarray(phase_initial)
                
            self.slm_amplitude = cp.ones(slm_size)
        else:
            self.target_amplitude = target_amplitude
            if weights is None:
                self.weights = np.ones_like(target_amplitude)
            else:
                self.weights = weights
                
            if phase_initial is None:
                self.phase = np.random.rand(*slm_size) * 2 * np.pi
            else:
                self.phase = phase_initial
                
            self.slm_amplitude = np.ones(slm_size)
        
        self.errors = []
    
    def forward_propagation(self, phase):
        """前向传播：使用CuPy的FFT"""
        slm_field = self.slm_amplitude * cp.exp(1j * phase)
        
        if self.use_gpu:
            # 使用CuPy的FFT
            focal_field = cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(slm_field)))
        else:
            focal_field = fftshift(fft2(ifftshift(cp.asnumpy(slm_field) if self.use_gpu else slm_field)))
            if self.use_gpu:
                focal_field = cp.asarray(focal_field)
        
        return focal_field
    
    def backward_propagation(self, focal_field):
        """反向传播：使用CuPy的逆FFT"""
        if self.use_gpu:
            slm_field = cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(focal_field)))
            phase = cp.angle(slm_field)
        else:
            slm_field = fftshift(ifft2(ifftshift(cp.asnumpy(focal_field) if self.use_gpu else focal_field)))
            phase = np.angle(slm_field)
            if self.use_gpu:
                phase = cp.asarray(phase)
        
        return phase
    
    def apply_constraints_focal_plane(self, focal_field):
        """在焦平面应用约束"""
        if self.use_gpu:
            current_amplitude = cp.abs(focal_field)
            current_phase = cp.angle(focal_field)
            new_amplitude = self.weights * self.target_amplitude + (1 - self.weights) * current_amplitude
            new_focal_field = new_amplitude * cp.exp(1j * current_phase)
        else:
            current_amplitude = np.abs(focal_field)
            current_phase = np.angle(focal_field)
            new_amplitude = self.weights * self.target_amplitude + (1 - self.weights) * current_amplitude
            new_focal_field = new_amplitude * np.exp(1j * current_phase)
        
        return new_focal_field
    
    def apply_constraints_slm_plane(self, slm_field):
        """在SLM平面应用约束"""
        # 提取相位，保持振幅不变（均匀照明）
        new_phase = np.angle(slm_field)
        
        return new_phase

    def calculate_error(self, focal_field):
        """计算误差"""
        if self.use_gpu:
            current_amplitude = cp.abs(focal_field)
            error = cp.sqrt(cp.sum((current_amplitude - self.target_amplitude)**2))
            error = cp.asnumpy(error)  # 转换为numpy用于存储
        else:
            current_amplitude = np.abs(focal_field)
            error = np.sqrt(np.sum((current_amplitude - self.target_amplitude)**2))
        
        self.errors.append(error)
        return error
    
    def run(self):
        """运行算法"""
        print("开始运行加权GS算法..." + ("使用GPU加速" if self.use_gpu else "使用CPU"))
        
        for i in range(self.iterations):
            focal_field = self.forward_propagation(self.phase)
            error = self.calculate_error(focal_field)
            constrained_focal_field = self.apply_constraints_focal_plane(focal_field)
            self.phase = self.backward_propagation(constrained_focal_field)
            
            if (i + 1) % 10 == 0:
                print(f"迭代 {i+1}/{self.iterations}, 误差: {error:.6f}")
        
        print("算法完成!")
        return self.get_phase_numpy(), self.errors
    
    def get_hologram(self):
        """获取全息图"""
        if self.use_gpu:
            hologram = cp.mod(self.phase, 2 * cp.pi)
            return cp.asnumpy(hologram)  # 转换回numpy用于显示
        else:
            return np.mod(self.phase, 2 * np.pi)
    
    def get_phase_numpy(self):
        """获取numpy格式的相位"""
        if self.use_gpu:
            return cp.asnumpy(self.phase)
        else:
            return self.phase

def create_target_amplitude(size, traps_positions, trap_std=2.0):
    """创建目标振幅分布"""
    amplitude = np.zeros(size)
    y, x = np.indices(size)
    
    for pos in traps_positions:
        gaussian = np.exp(-((x - pos[0])**2 + (y - pos[1])**2) / (2 * trap_std**2))
        amplitude += gaussian
    
    amplitude /= np.max(amplitude)
    return amplitude

    # ... [保持原有的 create_target_amplitude 函数代码不变] ...

# def create_weights(size, traps_positions, weight_value=0.98, background_weight=0.98):
#     """创建权重矩阵，确保值在[0,1]之间"""
#     weights = np.ones(size) * background_weight
#     y, x = np.indices(size)
    
#     for pos in traps_positions:
#         distance = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
#         # 使用高斯权重分布而不是硬阈值
#         gaussian_weight = background_weight + (weight_value - background_weight) * np.exp(-distance**2 / (2 * 3**2))
#         weights = np.maximum(weights, gaussian_weight)
    
#     return weights

def create_weights(size, traps_positions, weight_value=5.0, background_weight=0.98):
    """创建权重矩阵"""
    weights = np.ones(size) * background_weight
    y, x = np.indices(size)
    
    for pos in traps_positions:
        distance = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
        weights[distance < 5] = weight_value
    
    return weights

def generate_trap_configuration(slm_size, real_slm_size, grid_size=5, variation=0.3):
    """
    随机生成光镊阵列配置
    
    参数:
    variation: 位置随机变化的最大幅度（像素）
    """
    traps_positions = []
    base_spacing = real_slm_size[0] // (grid_size + 1)
    
    # 计算中心区域的起始位置
    center_offset_x = (slm_size[0] - real_slm_size[0]) // 2
    center_offset_y = (slm_size[1] - real_slm_size[1]) // 2
    
    for i in range(grid_size):
        for j in range(grid_size):
            # 基础位置
            base_x = center_offset_x + (j + 1) * base_spacing
            base_y = center_offset_y + (i + 1) * base_spacing
            
            # 添加随机变化
            if variation > 0:
                random_offset_x = np.random.randint(-variation, variation + 1)
                random_offset_y = np.random.randint(-variation, variation + 1)
                x = base_x + random_offset_x
                y = base_y + random_offset_y
            else:
                x, y = base_x, base_y
                
            traps_positions.append((x, y))
    
    return traps_positions

def save_training_data(dataset_path, data_dict, index):
    """保存训练数据到HDF5文件"""
    with h5py.File(dataset_path, 'a') as f:  # 'a' 模式表示追加
        group = f.create_group(f'sample_{index:04d}')
        for key, value in data_dict.items():
            group.create_dataset(key, data=value, compression='gzip')

def visualize_results(target_amplitude, hologram, reconstructed_amplitude, errors):
    """可视化结果"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # 目标振幅
    im1 = axes[0, 0].imshow(target_amplitude, cmap='hot')
    axes[0, 0].set_title('目标振幅分布')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 生成的全息图
    im2 = axes[0, 1].imshow(hologram, cmap='viridis')
    axes[0, 1].set_title('计算的全息图（相位）')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 重建的振幅
    im3 = axes[1, 0].imshow(reconstructed_amplitude, cmap='hot')
    axes[1, 0].set_title('重建的振幅分布')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 误差曲线
    axes[1, 1].plot(errors)
    axes[1, 1].set_title('迭代误差')
    axes[1, 1].set_xlabel('迭代次数')
    axes[1, 1].set_ylabel('误差')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def generate_training_samples(num_samples, output_dir, slm_size=(1024, 1024), 
                            real_slm_size=(256, 256), iterations=200, grid_size=5):
    """
    生成多组训练数据
    
    参数:
    num_samples: 要生成的样本数量
    output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    dataset_path = os.path.join(output_dir, 'training_data.h5')
    
    # 如果文件已存在，删除重建
    if os.path.exists(dataset_path):
        os.remove(dataset_path)
    
    total_time = 0
    successful_samples = 0
    
    for sample_idx in range(num_samples):
        print(f"\n{'='*50}")
        print(f"生成样本 {sample_idx + 1}/{num_samples}")
        print(f"{'='*50}")
        
        try:
            # 1. 随机生成光镊配置
            traps_positions = generate_trap_configuration(slm_size, real_slm_size, grid_size, variation=2)
            
            # 2. 创建目标振幅和权重
            target_amplitude = create_target_amplitude(slm_size, traps_positions, trap_std=2.5)
            weights = create_weights(slm_size, traps_positions, weight_value=0.98, background_weight=0.98)
            
            # 3. 运行GS算法
            start_time = time.time()
            
            wgs = WeightedGSAlgorithm(target_amplitude, weights, slm_size, iterations, use_gpu=True)
            phase, errors = wgs.run()
            
            # 4. 获取重建结果
            focal_field = wgs.forward_propagation(cp.asarray(phase))
            reconstructed_amplitude = cp.asnumpy(cp.abs(focal_field))
            hologram = wgs.get_hologram()
            
            # 5. 提取中心区域 (256x256)
            start_idx = (slm_size[0] - real_slm_size[0]) // 2
            end_idx = start_idx + real_slm_size[0]
            
            origin_target = target_amplitude[start_idx:end_idx, start_idx:end_idx]
            origin_reconstructed = reconstructed_amplitude[start_idx:end_idx, start_idx:end_idx]
            origin_phase = phase[start_idx:end_idx, start_idx:end_idx]
            origin_hologram = hologram[start_idx:end_idx, start_idx:end_idx]
            
            # 6. 计算保真度
            overlap = np.sum(origin_target * origin_reconstructed)
            fidelity = overlap / (np.sqrt(np.sum(origin_target**2)) * np.sqrt(np.sum(origin_reconstructed**2)))
            
            end_time = time.time()
            sample_time = end_time - start_time
            total_time += sample_time

            visualize_results(origin_target, origin_hologram, origin_reconstructed, errors)
            
            print(f"样本 {sample_idx + 1} 完成, 保真度: {fidelity:.4f}, 耗时: {sample_time:.2f}秒")
            
            # 7. 只保存高质量样本
            if fidelity > 0.85:  # 设置质量阈值
                # 准备数据字典
                sample_data = {
                    'target_amplitude': origin_target.astype(np.float32),
                    'optimal_phase': origin_phase.astype(np.float32),
                    'reconstructed_amplitude': origin_reconstructed.astype(np.float32),
                    'hologram': origin_hologram.astype(np.float32),
                    'fidelity': np.array([fidelity], dtype=np.float32),
                    'traps_positions': np.array(traps_positions, dtype=np.float32),
                    'errors': np.array(errors, dtype=np.float32)
                }
                
                # 保存数据
                save_training_data(dataset_path, sample_data, successful_samples)
                successful_samples += 1
                
                # 每10个样本保存一次预览图
                if successful_samples % 10 == 0:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    axes[0].imshow(origin_target, cmap='hot')
                    axes[0].set_title(f'Target (Fidelity: {fidelity:.3f})')
                    axes[1].imshow(origin_reconstructed, cmap='hot')
                    axes[1].set_title('Reconstructed')
                    plt.savefig(os.path.join(output_dir, f'preview_{successful_samples:04d}.png'))
                    plt.close()
                    
            else:
                print(f"样本 {sample_idx + 1} 保真度太低 ({fidelity:.4f}), 跳过保存")
                
        except Exception as e:
            print(f"生成样本 {sample_idx + 1} 时出错: {str(e)}")
            continue
    
    # 生成总结报告
    print(f"\n{'='*50}")
    print("数据生成完成!")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"成功样本: {successful_samples}/{num_samples}")
    print(f"平均每样本耗时: {total_time/max(1, successful_samples):.2f}秒")
    print(f"数据保存至: {dataset_path}")
    
    # 保存数据集信息
    dataset_info = {
        'num_samples': successful_samples,
        'slm_size': slm_size,
        'real_slm_size': real_slm_size,
        'grid_size': grid_size,
        'creation_date': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with h5py.File(dataset_path, 'a') as f:
        f.attrs.update(dataset_info)

# 示例使用
if __name__ == "__main__":
    # 配置参数
    NUM_SAMPLES = 100      # 想要生成的总样本数
    ITERATIONS = 50000        # 每个样本的迭代次数（可以比50000少，200-500通常足够收敛）
    OUTPUT_DIR = "./gs_training_data_01"  # 输出目录
    
    # 开始生成数据
    generate_training_samples(
        num_samples=NUM_SAMPLES,
        output_dir=OUTPUT_DIR,
        slm_size=(1024, 1024),
        real_slm_size=(256, 256),
        iterations=ITERATIONS,
        grid_size=5
    )

    