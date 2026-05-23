# -*- coding: utf-8 -*-
"""
GS_16_final_gen.py (最终生产版)

功能:
1. 使用【全复振幅约束 WGS】生成高质量的 (Input, Label) 数据对。
2. 包含 Amplitude 和 Phase 的精确控制。
3. 保存为 HDF5 格式供 AI 训练。

数据流:
Input (Target) -> WGS (Complex Constraint) -> Hologram -> FFT -> Label (Physical Field)
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import os
import time

# 设置显卡
cp.cuda.Device(0).use()

# Matplotlib 中文设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 核心算法类 (已验证通过的 GS_15 版本)
# ==========================================
class WeightedGSAlgorithm:
    def __init__(self, target_amplitude, target_phase, weights=None, slm_size=(1024, 1024), iterations=50, phase_initial=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.slm_size = slm_size
        self.iterations = iterations

        if use_gpu:
            self.xp = cp
        else:
            self.xp = np

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
        # 1. 归一化 Current Field
        current_amp = self.xp.abs(focal_field)
        max_val = self.xp.max(current_amp)
        factor = 1.0 / (max_val + 1e-10)
        current_field_norm = focal_field * factor

        # 2. 复数混合 (Complex Mixing)
        new_focal_field = self.weights * self.target_field + (1 - self.weights) * current_field_norm
        
        return new_focal_field

    def calculate_error(self, focal_field):
        current_amp = self.xp.abs(focal_field)
        factor = 1.0 / (self.xp.max(current_amp) + 1e-10)
        current_field_norm = focal_field * factor
        
        diff = self.target_field - current_field_norm
        error = self.xp.sqrt(self.xp.sum(self.xp.abs(diff)**2))
        
        if self.use_gpu:
            self.errors.append(float(error))
        else:
            self.errors.append(error)
        return error
    
    def update_adaptive_weights(self, focal_field):
        """
        自适应调整权重，强制所有光斑亮度一致。
        """
        # 1. 获取当前焦平面振幅
        current_amp = self.xp.abs(focal_field)
        
        # 2. 提取每个光斑位置的亮度 (Peak Intensity)
        # 注意：这里需要传入光斑坐标 traps_positions 才能精确提取
        # 为了简化，我们可以利用 mask (目标振幅 > 0 的区域)
        mask = self.target_amplitude > 0.1
        
        # 简单粗暴版：直接用 target / current 的比值来修正
        # 如果 current 小，比值 > 1，权重增加；反之减少。
        # 加上一个小常数防止除以零
        correction = self.target_amplitude / (current_amp + 1e-6)
        
        # 只更新光斑区域的权重，背景权重保持不变
        # 引入一个学习率 alpha (比如 0.1)，避免震荡
        alpha = 0.1
        new_weights = self.weights * (1 - alpha + alpha * correction)
        
        # 更新权重 (保持背景权重不动，防止背景爆亮)
        self.weights[mask] = new_weights[mask]

    def run(self):
        iterator = tqdm(range(self.iterations), desc="WGS Iteration", leave=False)
        for i in iterator:
            focal_field = self.forward_propagation(self.phase)
            self.calculate_error(focal_field)
            
            # === 新增：每隔几次迭代，更新一次权重 ===
            # 不要每次都更，容易震荡。前10次不更，让它先收敛一点。
            if i > 10 and i % 2 == 0: 
                self.update_adaptive_weights(focal_field)
            
            constrained_focal_field = self.apply_constraints_focal_plane(focal_field)
            self.phase = self.backward_propagation(constrained_focal_field)
            
        return self.get_phase_numpy(), self.errors

    # def run(self):
    #     iterator = tqdm(range(self.iterations), desc="WGS Iteration", leave=False)
    #     for _ in iterator:
    #         focal_field = self.forward_propagation(self.phase)
    #         self.calculate_error(focal_field)
    #         constrained_focal_field = self.apply_constraints_focal_plane(focal_field)
    #         self.phase = self.backward_propagation(constrained_focal_field)
        
    #     if self.use_gpu:
    #         return cp.asnumpy(self.phase), self.errors
    #     else:
    #         return self.phase, self.errors
        
    def get_focal_field(self, phase):
        # 辅助函数：方便获取最终的焦平面场
        phase_gpu = self.xp.asarray(phase)
        return self.forward_propagation(phase_gpu)

# ==========================================
# 2. 数据生成辅助函数
# ==========================================
def generate_trap_configuration(slm_size, real_slm_size, min_traps=5, max_traps=50):
    num_traps = np.random.randint(min_traps, max_traps + 1)
    traps_positions = []
    traps_phases = []

    center_offset_x = (slm_size[0] - real_slm_size[0]) // 2
    center_offset_y = (slm_size[1] - real_slm_size[1]) // 2

    for _ in range(num_traps):
        x = center_offset_x + np.random.randint(0, real_slm_size[0])
        y = center_offset_y + np.random.randint(0, real_slm_size[1])
        traps_positions.append((x, y))
        # 随机相位 [-pi, pi]
        phase = np.random.uniform(-np.pi, np.pi)
        traps_phases.append(phase)

    return traps_positions, traps_phases

def create_target_amplitude(size, traps_positions, trap_std=2.0):
    amplitude = np.zeros(size, dtype=np.float32)
    y_idx, x_idx = np.indices(size)
    for pos in traps_positions:
        # 使用较小的 trap_std 生成锐利的目标
        gaussian = np.exp(-((x_idx - pos[0])**2 + (y_idx - pos[1])**2) / (2 * trap_std**2))
        amplitude = np.maximum(amplitude, gaussian)
    return amplitude

def create_input_phase_map(size, traps_positions, traps_phases, trap_std=2.0):
    """
    为输入生成稠密的相位图 (仅在光镊位置有值，并做小范围高斯扩散方便CNN读取)
    """
    phase_map = np.zeros(size, dtype=np.float32)
    y_idx, x_idx = np.indices(size)
    
    for pos, phase in zip(traps_positions, traps_phases):
        # 半径 3 个像素内的点都赋值为该相位
        dist_sq = (x_idx - pos[0])**2 + (y_idx - pos[1])**2
        mask = dist_sq < (trap_std * 2)**2 
        phase_map[mask] = phase
        
    return phase_map

def create_weights(size, traps_positions, weight_value=20.0): 
    # 权重矩阵：光镊处权重极高，强迫算法优先满足光镊处的振幅和相位
    weights = np.ones(size, dtype=np.float32) * 0.1 # 背景权重 0.1
    y_idx, x_idx = np.indices(size)
    for pos in traps_positions:
        mask = ((x_idx - pos[0])**2 + (y_idx - pos[1])**2) < 25 # 半径5
        weights[mask] = weight_value
    return weights

def save_training_data(dataset_path, data_dict, index):
    with h5py.File(dataset_path, 'a') as f:
        group = f.create_group(f'sample_{index:05d}')
        for key, value in data_dict.items():
            group.create_dataset(key, data=value, compression='gzip')

# ==========================================
# 3. 主流程
# ==========================================
def generate_training_samples(num_samples, output_dir, slm_size=(1024, 1024), real_slm_size=(256, 256), iterations=1000):
    os.makedirs(output_dir, exist_ok=True)
    dataset_path = os.path.join(output_dir, 'training_data_complex.h5')

    if os.path.exists(dataset_path):
        os.remove(dataset_path)

    # 裁剪区域定义
    start_idx_h = (slm_size[0] - real_slm_size[0]) // 2
    end_idx_h = start_idx_h + real_slm_size[0]
    start_idx_w = (slm_size[1] - real_slm_size[1]) // 2
    end_idx_w = start_idx_w + real_slm_size[1]

    successful_samples = 0
    pbar = tqdm(range(num_samples))

    for sample_idx in pbar:
        try:
            # 1. 随机生成配置 (包含相位!)
            traps_pos, traps_phi = generate_trap_configuration(slm_size, real_slm_size)
            
            # 2. 准备 WGS 输入
            input_amp_full = create_target_amplitude(slm_size, traps_pos, trap_std=2.0)
            input_phase_full = create_input_phase_map(slm_size, traps_pos, traps_phi, trap_std=2.0)
            weights_full = create_weights(slm_size, traps_pos, weight_value=20.0) # 强权重
            
            # 3. 运行 WGS (Complex Constraint)
            # 注意: 传入 input_phase_full 作为目标相位
            wgs = WeightedGSAlgorithm(input_amp_full, input_phase_full, weights_full, slm_size, iterations, use_gpu=True)
            final_slm_phase, errors = wgs.run()

            # 4. 计算 Label (Ground Truth)
            # 将 WGS 算出的全息图反推回焦平面，得到"物理上可实现的"振幅和相位
            focal_field_complex = wgs.get_focal_field(final_slm_phase)
            
            label_amp_full = cp.asnumpy(cp.abs(focal_field_complex))
            label_phase_full = cp.asnumpy(cp.angle(focal_field_complex))
            
            # 5. 准备 CNN 输入和输出 (裁剪)
            def crop(arr): return arr[start_idx_h:end_idx_h, start_idx_w:end_idx_w]
            
            # 归一化 Label 振幅 (防止量级过大)
            max_val = np.max(label_amp_full)
            norm_factor = max_val if max_val > 1e-8 else 1.0
            
            sample_data = {
                # 输入: 理想的振幅和相位
                'input_amplitude': crop(input_amp_full).astype(np.float32),
                'input_phase': crop(input_phase_full).astype(np.float32),
                
                # 标签: WGS 算出的物理场 (振幅归一化，相位保持原始值)
                'output_amplitude': (crop(label_amp_full) / norm_factor).astype(np.float32), 
                'output_phase': crop(label_phase_full).astype(np.float32),
                
                'num_traps': np.array([len(traps_pos)])
            }
            
            save_training_data(dataset_path, sample_data, successful_samples)
            successful_samples += 1
            pbar.set_description(f"Saved: {successful_samples}")

        except Exception as e:
            print(f"Error: {e}")
            continue

    print(f"Done. Saved to {dataset_path}")

if __name__ == "__main__":
    # 建议先生成 1000 个样本
    generate_training_samples(1000, "./ustc/Others/ai原子阵列重排/gs_training_data/gs_training_data_05")