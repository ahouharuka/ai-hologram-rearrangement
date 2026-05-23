import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import ndimage
from matplotlib import font_manager
import time


plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False



class WeightedGSAlgorithm:
    def __init__(self, target_amplitude, weights=None, slm_size=(512, 512), 
                 iterations=50, phase_initial=None):
        """
        初始化加权GS算法
        
        参数:
        target_amplitude: 目标振幅分布（光镊阵列的强度分布）
        weights: 权重矩阵，用于强调特定区域的重要性
        slm_size: SLM的尺寸（像素）
        iterations: 迭代次数
        phase_initial: 初始相位估计（如果为None，则使用随机相位）
        """
        self.target_amplitude = target_amplitude
        self.slm_size = slm_size
        self.iterations = iterations
        
        # 如果没有提供权重，默认使用目标振幅作为权重
        if weights is None:
            self.weights = np.ones_like(target_amplitude)
        else:
            self.weights = weights
            
        # 初始化相位
        if phase_initial is None:
            self.phase = np.random.rand(*slm_size) * 2 * np.pi
        else:
            self.phase = phase_initial
            
        # 初始化SLM平面振幅（通常假设为均匀照明）
        self.slm_amplitude = np.ones(slm_size)
        
        # 记录每次迭代的误差
        self.errors = []
    
    def forward_propagation(self, phase):
        """前向传播：从SLM平面到焦平面"""
        # 创建SLM平面的复光场
        slm_field = self.slm_amplitude * np.exp(1j * phase)
        
        # 傅里叶变换到焦平面
        focal_field = fftshift(fft2(ifftshift(slm_field)))
        
        return focal_field
    
    def backward_propagation(self, focal_field):
        """反向传播：从焦平面回到SLM平面"""
        # 逆傅里叶变换回SLM平面
        slm_field = fftshift(ifft2(ifftshift(focal_field)))
        
        # 提取相位
        phase = np.angle(slm_field)
        
        return phase
    
    def apply_constraints_focal_plane(self, focal_field):
        """在焦平面应用约束"""
        # 计算当前焦平面的振幅和相位
        current_amplitude = np.abs(focal_field)
        current_phase = np.angle(focal_field)
        
        # 应用加权约束：将振幅替换为目标振幅，但保留计算的相位
        # 加权GS的关键步骤：使用权重来调整更新
        new_amplitude = self.weights * self.target_amplitude + (1 - self.weights) * current_amplitude
        
        # 创建新的焦平面光场
        new_focal_field = new_amplitude * np.exp(1j * current_phase)
        
        return new_focal_field
    
    def apply_constraints_slm_plane(self, slm_field):
        """在SLM平面应用约束"""
        # 提取相位，保持振幅不变（均匀照明）
        new_phase = np.angle(slm_field)
        
        return new_phase
    
    def calculate_error(self, focal_field):
        """计算当前迭代的误差"""
        current_amplitude = np.abs(focal_field)
        error = np.sqrt(np.sum((current_amplitude - self.target_amplitude)**2))
        self.errors.append(error)
        return error
    
    def run(self):
        """运行加权GS算法"""
        print("开始运行加权GS算法...")
        
        for i in range(self.iterations):
            # 前向传播：SLM平面 -> 焦平面
            focal_field = self.forward_propagation(self.phase)
            
            # 计算并记录误差
            error = self.calculate_error(focal_field)
            
            # 在焦平面应用约束
            constrained_focal_field = self.apply_constraints_focal_plane(focal_field)
            
            # 反向传播：焦平面 -> SLM平面
            self.phase = self.backward_propagation(constrained_focal_field)
            
            # 在SLM平面应用约束
            # self.phase = self.apply_constraints_slm_plane(slm_field)
            
            if (i + 1) % 10 == 0:
                print(f"迭代 {i+1}/{self.iterations}, 误差: {error:.6f}")
        
        print("算法完成!")
        return self.phase, self.errors
    
    def get_hologram(self):
        """获取最终的全息图（SLM需要的相位图案）"""
        # 将相位范围从[-π, π]调整到[0, 2π]
        hologram = np.mod(self.phase, 2 * np.pi)
        return hologram

def create_target_amplitude(size, traps_positions, trap_std=2.0):
    """
    创建目标振幅分布（多个高斯光斑）
    
    参数:
    size: 图像尺寸
    traps_positions: 光镊位置列表 [(x1, y1), (x2, y2), ...]
    trap_std: 高斯光斑的标准差（控制光斑大小）
    """
    amplitude = np.zeros(size)
    y, x = np.indices(size)
    
    for pos in traps_positions:
        # 创建高斯分布
        gaussian = np.exp(-((x - pos[0])**2 + (y - pos[1])**2) / (2 * trap_std**2))
        amplitude += gaussian
    
    # 归一化到[0, 1]范围
    amplitude /= np.max(amplitude)
    
    return amplitude

def create_weights(size, traps_positions, weight_value=5.0, background_weight=0.98):
    """
    创建权重矩阵，强调光镊位置的重要性
    
    参数:
    size: 图像尺寸
    traps_positions: 光镊位置列表
    weight_value: 光镊位置的权重值
    background_weight: 背景区域的权重值
    """
    weights = np.ones(size) * background_weight
    y, x = np.indices(size)
    
    for pos in traps_positions:
        # 在光镊位置创建高权重区域
        distance = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
        weights[distance < 5] = weight_value  # 5像素半径内设置为高权重
    
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
    slm_size = (1024, 1024)  # SLM分辨率
    iterations = 1000  # 迭代次数
    real_slm_size = (256, 256)
    
    # 创建目标光镊阵列（例如5x5网格）
    traps_positions = []
    grid_size = 5
    spacing = real_slm_size[0] // (grid_size + 1)
    
    for i in range(grid_size):
        for j in range(grid_size):
            x = (slm_size[0] - real_slm_size[0])/2 + (j + 1) * spacing
            y = (slm_size[1] - real_slm_size[1])/2 + (i + 1) * spacing
            traps_positions.append((x, y))
    
    # 创建目标振幅分布
    target_amplitude = create_target_amplitude(slm_size, traps_positions, trap_std=3.0)
    
    # 创建权重矩阵（强调光镊位置）
    weights = create_weights(slm_size, traps_positions, weight_value=0.98)
    
    # 初始化并运行加权GS算法
    tm1 = time.time()

    wgs = WeightedGSAlgorithm(target_amplitude, weights, slm_size, iterations)
    phase, errors = wgs.run()

    tm2 = time.time()
    print("Time:", tm2-tm1)
    
    # 获取全息图
    hologram = wgs.get_hologram()
    
    # 验证全息图效果：使用全息图进行前向传播
    focal_field = wgs.forward_propagation(phase)
    reconstructed_amplitude = np.abs(focal_field)
    
    # 可视化结果
    visualize_results(target_amplitude, hologram, reconstructed_amplitude, errors)
    
    # 计算并打印保真度
    overlap = np.sum(target_amplitude * reconstructed_amplitude)
    fidelity = overlap / (np.sqrt(np.sum(target_amplitude**2)) * np.sqrt(np.sum(reconstructed_amplitude**2)))
    print(f"重建保真度: {fidelity:.4f}")
    