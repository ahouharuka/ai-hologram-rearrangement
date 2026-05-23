import numpy as np
import cupy as cp  # 导入CuPy
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import ndimage
from matplotlib import font_manager
import time

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

# 修改工具函数，确保返回numpy数组
def create_target_amplitude(size, traps_positions, trap_std=2.0):
    """创建目标振幅分布"""
    amplitude = np.zeros(size)
    y, x = np.indices(size)
    
    for pos in traps_positions:
        gaussian = np.exp(-((x - pos[0])**2 + (y - pos[1])**2) / (2 * trap_std**2))
        amplitude += gaussian
    
    amplitude /= np.max(amplitude)
    return amplitude

def create_weights(size, traps_positions, weight_value=5.0, background_weight=0.98):
    """创建权重矩阵"""
    weights = np.ones(size) * background_weight
    y, x = np.indices(size)
    
    for pos in traps_positions:
        distance = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
        weights[distance < 5] = weight_value
    
    return weights

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

# 示例使用
if __name__ == "__main__":
    # 设置参数
    slm_size = (1024, 1024)
    iterations = 50000
    real_slm_size = (256, 256)
    
    # 创建目标光镊阵列
    traps_positions = []
    grid_size = 5
    spacing = real_slm_size[0] // (grid_size + 1)
    
    for i in range(grid_size):
        for j in range(grid_size):
            x = (slm_size[0] - real_slm_size[0])/2 + (j + 1) * spacing
            y = (slm_size[1] - real_slm_size[1])/2 + (i + 1) * spacing
            traps_positions.append((x, y))
    
    # 创建目标振幅和权重
    target_amplitude = create_target_amplitude(slm_size, traps_positions, trap_std=3.0)
    weights = create_weights(slm_size, traps_positions, weight_value=0.98)
    
    # 使用GPU运行
    tm1 = time.time()
    
    wgs = WeightedGSAlgorithm(target_amplitude, weights, slm_size, iterations, use_gpu=True)
    phase, errors = wgs.run()
    
    tm2 = time.time()
    print(f"GPU运行时间: {tm2-tm1:.2f}秒")
    
    # 获取结果
    hologram = wgs.get_hologram()
    focal_field = wgs.forward_propagation(cp.asarray(phase) if wgs.use_gpu else phase)
    
    if wgs.use_gpu:
        reconstructed_amplitude = cp.asnumpy(cp.abs(focal_field))
    else:
        reconstructed_amplitude = np.abs(focal_field)


    start_idx = (slm_size[0] - real_slm_size[0]) // 2
    end_idx = start_idx + real_slm_size[0]

    origin_target_amplitude = target_amplitude[start_idx:end_idx, start_idx:end_idx]
    origin_reconstructed_amplitude = reconstructed_amplitude[start_idx:end_idx, start_idx:end_idx]
    origin_phase = phase[start_idx:end_idx, start_idx:end_idx]
    origin_hologram = hologram[start_idx:end_idx, start_idx:end_idx]
    
    # 可视化结果
    # visualize_results(target_amplitude, hologram, reconstructed_amplitude, errors)
    visualize_results(origin_target_amplitude, origin_hologram, origin_reconstructed_amplitude, errors)
    
    # 计算保真度
    overlap = np.sum(target_amplitude * reconstructed_amplitude)
    fidelity = overlap / (np.sqrt(np.sum(target_amplitude**2)) * np.sqrt(np.sum(reconstructed_amplitude**2)))
    print(f"重建保真度: {fidelity:.4f}")

    real_overlap = np.sum(origin_target_amplitude * origin_reconstructed_amplitude)
    real_fidelity = real_overlap / (np.sqrt(np.sum(origin_target_amplitude**2)) * np.sqrt(np.sum(origin_reconstructed_amplitude**2)))
    print(f"重建中心保真度: {real_fidelity:.4f}")