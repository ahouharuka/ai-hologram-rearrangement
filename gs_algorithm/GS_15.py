# -*- coding: utf-8 -*-
"""
GS_15_ComplexConstraint.py

修正重点:
1. 实现了【全复振幅约束】(Complex Field Constraint)。
2. WGS 现在会同时优化振幅和相位，强迫焦平面相位接近 input_phase。
3. 这才是论文中 AI 能学习 "Position and Phase" 的根本原因。
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm

# Matplotlib 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class WeightedGSAlgorithm:
    def __init__(self, target_amplitude, target_phase, weights=None, slm_size=(1024, 1024), iterations=50, phase_initial=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.slm_size = slm_size
        self.iterations = iterations

        if use_gpu:
            self.xp = cp
        else:
            self.xp = np

        # === 关键变化：我们现在有了目标相位 target_phase ===
        self.target_amplitude = self.xp.asarray(target_amplitude)
        self.target_phase = self.xp.asarray(target_phase)
        
        # 构建目标复数场 (Ideal Target Field)
        self.target_field = self.target_amplitude * self.xp.exp(1j * self.target_phase)

        self.weights = self.xp.ones_like(self.target_amplitude) if weights is None else self.xp.asarray(weights)
        
        if phase_initial is None:
            self.phase = self.xp.random.rand(*slm_size) * 2 * self.xp.pi
        else:
            self.phase = self.xp.asarray(phase_initial)
            
        self.slm_amplitude = self.xp.ones(slm_size)
        self.errors = []

    def forward_propagation(self, phase):
        slm_field = self.slm_amplitude * self.xp.exp(1j * phase)
        focal_field = self.xp.fft.fftshift(self.xp.fft.fft2(self.xp.fft.ifftshift(slm_field)))
        return focal_field

    def backward_propagation(self, focal_field):
        slm_field = self.xp.fft.fftshift(self.xp.fft.ifft2(self.xp.fft.ifftshift(focal_field)))
        phase = self.xp.angle(slm_field)
        return phase

    def apply_constraints_focal_plane(self, focal_field):
        # === 核心修正：复数域混合 (Complex Mixing) ===
        # 我们不再只混合振幅，而是混合整个复数向量。
        # 公式: New_Field = Weights * Target_Field + (1 - Weights) * Current_Field
        
        # 1. 归一化 Current Field (防止量级爆炸问题)
        current_amp = self.xp.abs(focal_field)
        max_val = self.xp.max(current_amp)
        if max_val > 1e-10:
            factor = 1.0 / max_val
        else:
            factor = 1.0
        
        current_field_norm = focal_field * factor

        # 2. 复数混合
        # self.target_field 包含了我们想要的相位信息！
        # 这一步会强迫优化方向朝向 "特定的相位"
        new_focal_field = self.weights * self.target_field + (1 - self.weights) * current_field_norm
        
        return new_focal_field

    def calculate_error(self, focal_field):
        # 误差计算也应该考虑相位了 (复数距离)
        # 归一化 current
        current_amp = self.xp.abs(focal_field)
        factor = 1.0 / (self.xp.max(current_amp) + 1e-10)
        current_field_norm = focal_field * factor
        
        # 计算 |Target - Current|^2
        diff = self.target_field - current_field_norm
        error = self.xp.sqrt(self.xp.sum(self.xp.abs(diff)**2))
        
        if self.use_gpu:
            self.errors.append(float(error))
        else:
            self.errors.append(error)
        return error

    def run(self):
        iterator = tqdm(range(self.iterations), desc="WGS Iteration", leave=False)
        for _ in iterator:
            focal_field = self.forward_propagation(self.phase)
            self.calculate_error(focal_field)
            constrained_focal_field = self.apply_constraints_focal_plane(focal_field)
            self.phase = self.backward_propagation(constrained_focal_field)
        
        if self.use_gpu:
            return cp.asnumpy(self.phase), self.errors
        else:
            return self.phase, self.errors

# === 辅助函数 ===
def create_target_amplitude(size, traps_positions, trap_std=2.0):
    amplitude = np.zeros(size, dtype=np.float32)
    y_idx, x_idx = np.indices(size)
    for pos in traps_positions:
        gaussian = np.exp(-((x_idx - pos[0])**2 + (y_idx - pos[1])**2) / (2 * trap_std**2))
        amplitude = np.maximum(amplitude, gaussian)
    return amplitude

def create_weights(size, traps_positions, weight_value=20.0): 
    # 稍微调高权重，强制相位对齐
    weights = np.ones(size, dtype=np.float32) * 0.1
    y_idx, x_idx = np.indices(size)
    for pos in traps_positions:
        mask = ((x_idx - pos[0])**2 + (y_idx - pos[1])**2) < 25
        weights[mask] = weight_value
    return weights

def debug_wgs():
    print("正在测试全复振幅约束 WGS...")
    slm_size = (1024, 1024)
    
    # 1. 定义光镊位置
    traps_pos = [(512, 512), (400, 400)]
    
    # 2. 定义目标相位 (Target Phase)
    # 我们故意让两个点的相位不一样，看看WGS能不能学会
    # 点1相位: 0, 点2相位: pi/2 (1.57)
    input_amp = create_target_amplitude(slm_size, traps_pos)
    input_phase = np.zeros(slm_size, dtype=np.float32)
    
    # 手动设置相位图
    y_idx, x_idx = np.indices(slm_size)
    # Point 1: (512, 512) -> Phase 0
    # Point 2: (400, 400) -> Phase pi/2
    mask1 = ((x_idx - 512)**2 + (y_idx - 512)**2) < 25
    mask2 = ((x_idx - 400)**2 + (y_idx - 400)**2) < 25
    input_phase[mask1] = 0.0
    input_phase[mask2] = 1.57 # 强行要求这个点相位是 1.57
    
    weights = create_weights(slm_size, traps_pos)
    
    # 3. 运行 WGS (传入 target_phase!)
    wgs = WeightedGSAlgorithm(input_amp, input_phase, weights, slm_size, iterations=1000, use_gpu=True)
    final_phase_slm, errors = wgs.run()
    
    # 4. 验证结果
    phase_input_gpu = cp.asarray(final_phase_slm)
    focal_field = wgs.forward_propagation(phase_input_gpu)
    
    recon_amp = cp.asnumpy(cp.abs(focal_field))
    recon_phase = cp.asnumpy(cp.angle(focal_field))
    
    # 5. 检查相位是否对齐
    p1_phase = recon_phase[512, 512]
    p2_phase = recon_phase[400, 400]
    
    print(f"目标相位: Point1=0.0, Point2=1.57")
    print(f"实际WGS生成相位: Point1={p1_phase:.3f}, Point2={p2_phase:.3f}")
    
    # 画图
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Recon Amp")
    plt.imshow(recon_amp[350:600, 350:600], cmap='hot', vmin=0, vmax=recon_amp.max())
    
    plt.subplot(1, 2, 2)
    plt.title("Recon Phase")
    plt.imshow(recon_phase[350:600, 350:600], cmap='hsv', vmin=-3.14, vmax=3.14)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    debug_wgs()