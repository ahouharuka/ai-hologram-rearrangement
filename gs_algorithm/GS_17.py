# -*- coding: utf-8 -*-
"""
GS_17_final_gen_vis.py (带可视化监控版)

功能:
1. 生成高质量全复振幅约束数据。
2. 【新增】每生成10组数据，自动保存一张 Loss 曲线和重建效果图，方便监控 WGS 质量。
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
# 1. 核心算法类 (GS_15 版本)
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
        
        # 构建目标复数场
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
        current_amp = self.xp.abs(focal_field)
        max_val = self.xp.max(current_amp)
        factor = 1.0 / (max_val + 1e-10)
        current_field_norm = focal_field * factor

        # 复数混合
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
        """自适应权重更新 (可选功能)"""
        current_amp = self.xp.abs(focal_field)
        mask = self.target_amplitude > 0.1
        correction = self.target_amplitude / (current_amp + 1e-6)
        alpha = 0.1
        new_weights = self.weights * (1 - alpha + alpha * correction)
        self.weights[mask] = new_weights[mask]

    def run(self):
        iterator = tqdm(range(self.iterations), desc="WGS Iteration", leave=False)
        for i in iterator:
            focal_field = self.forward_propagation(self.phase)
            self.calculate_error(focal_field)
            
            # 简单的自适应权重更新策略
            if i > 10 and i % 2 == 0: 
                self.update_adaptive_weights(focal_field)

            constrained_focal_field = self.apply_constraints_focal_plane(focal_field)
            self.phase = self.backward_propagation(constrained_focal_field)
        
        if self.use_gpu:
            return cp.asnumpy(self.phase), self.errors
        else:
            return self.phase, self.errors
        
    def get_focal_field(self, phase):
        phase_gpu = self.xp.asarray(phase)
        return self.forward_propagation(phase_gpu)

# ==========================================
# 2. 辅助函数
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
        phase = np.random.uniform(-np.pi, np.pi)
        traps_phases.append(phase)
    return traps_positions, traps_phases

def create_target_amplitude(size, traps_positions, trap_std=2.0):
    amplitude = np.zeros(size, dtype=np.float32)
    y_idx, x_idx = np.indices(size)
    for pos in traps_positions:
        gaussian = np.exp(-((x_idx - pos[0])**2 + (y_idx - pos[1])**2) / (2 * trap_std**2))
        amplitude = np.maximum(amplitude, gaussian)
    return amplitude

def create_input_phase_map(size, traps_positions, traps_phases, trap_std=2.0):
    phase_map = np.zeros(size, dtype=np.float32)
    y_idx, x_idx = np.indices(size)
    for pos, phase in zip(traps_positions, traps_phases):
        dist_sq = (x_idx - pos[0])**2 + (y_idx - pos[1])**2
        mask = dist_sq < (trap_std * 2)**2 
        phase_map[mask] = phase
    return phase_map

def create_weights(size, traps_positions, weight_value=50.0): 
    weights = np.ones(size, dtype=np.float32) * 0.1 
    y_idx, x_idx = np.indices(size)
    for pos in traps_positions:
        mask = ((x_idx - pos[0])**2 + (y_idx - pos[1])**2) < 25 
        weights[mask] = weight_value
    return weights

def save_training_data(dataset_path, data_dict, index):
    with h5py.File(dataset_path, 'a') as f:
        group = f.create_group(f'sample_{index:05d}')
        for key, value in data_dict.items():
            group.create_dataset(key, data=value, compression='gzip')

# ==========================================
# 3. 新增：可视化函数
# ==========================================
def visualize_sample(save_dir, index, input_amp, recon_amp, recon_phase, errors, crop_coords):
    """
    保存单个样本的可视化结果
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 为了画图清晰，我们把 numpy 数据拉回 CPU
    # recon_amp 和 input_amp 已经是 CPU numpy 了
    
    # 裁剪区域 (用于标题显示范围)
    h_start, h_end, w_start, w_end = crop_coords
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Sample {index:05d} Analysis', fontsize=16)
    
    # 1. Input Target (Ideal)
    axes[0, 0].imshow(input_amp, cmap='hot', vmin=0, vmax=1)
    axes[0, 0].set_title('Input Target (Ideal)')
    
    # 2. WGS Reconstruction
    im2 = axes[0, 1].imshow(recon_amp, cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title('WGS Reconstruction (Actual)')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    # 3. WGS Phase
    im3 = axes[1, 0].imshow(recon_phase, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[1, 0].set_title('Reconstructed Phase')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    # 4. Error Curve
    axes[1, 1].plot(errors, 'b-')
    axes[1, 1].set_title('WGS Convergence (Error)')
    axes[1, 1].set_xlabel('Iterations')
    axes[1, 1].set_ylabel('Error')
    axes[1, 1].set_yscale('log') # 使用对数坐标看收敛细节
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'vis_sample_{index:05d}.png')
    plt.savefig(save_path)
    plt.close() # 关闭图片释放内存

# ==========================================
# 4. 主流程
# ==========================================
def generate_training_samples(num_samples, output_dir, slm_size=(1024, 1024), real_slm_size=(256, 256), iterations=1000):
    os.makedirs(output_dir, exist_ok=True)
    dataset_path = os.path.join(output_dir, 'training_data_complex.h5')
    
    # 预览图保存文件夹
    preview_dir = os.path.join(output_dir, 'preview_images')

    if os.path.exists(dataset_path):
        os.remove(dataset_path)

    # 裁剪区域
    start_idx_h = (slm_size[0] - real_slm_size[0]) // 2
    end_idx_h = start_idx_h + real_slm_size[0]
    start_idx_w = (slm_size[1] - real_slm_size[1]) // 2
    end_idx_w = start_idx_w + real_slm_size[1]

    successful_samples = 0
    pbar = tqdm(range(num_samples))

    for sample_idx in pbar:
        try:
            # 1. 生成配置
            traps_pos, traps_phi = generate_trap_configuration(slm_size, real_slm_size)
            
            # 2. 准备 WGS 输入
            input_amp_full = create_target_amplitude(slm_size, traps_pos, trap_std=2.0)
            input_phase_full = create_input_phase_map(slm_size, traps_pos, traps_phi, trap_std=2.0)
            weights_full = create_weights(slm_size, traps_pos, weight_value=20.0)
            
            # 3. 运行 WGS
            wgs = WeightedGSAlgorithm(input_amp_full, input_phase_full, weights_full, slm_size, iterations, use_gpu=True)
            final_slm_phase, errors = wgs.run()

            # 4. 计算 Label
            focal_field_complex = wgs.get_focal_field(final_slm_phase)
            label_amp_full = cp.asnumpy(cp.abs(focal_field_complex))
            label_phase_full = cp.asnumpy(cp.angle(focal_field_complex))
            
            # 5. 裁剪数据
            def crop(arr): return arr[start_idx_h:end_idx_h, start_idx_w:end_idx_w]
            
            max_val = np.max(label_amp_full)
            norm_factor = max_val if max_val > 1e-8 else 1.0
            
            input_amp_crop = crop(input_amp_full).astype(np.float32)
            output_amp_crop = (crop(label_amp_full) / norm_factor).astype(np.float32)
            output_phase_crop = crop(label_phase_full).astype(np.float32)
            
            sample_data = {
                'input_amplitude': input_amp_crop,
                'input_phase': crop(input_phase_full).astype(np.float32),
                'output_amplitude': output_amp_crop, 
                'output_phase': output_phase_crop,
                'num_traps': np.array([len(traps_pos)])
            }
            
            save_training_data(dataset_path, sample_data, successful_samples)
            
            # === 新增：每 10 个样本保存一次可视化结果 ===
            if successful_samples % 10 == 0:
                visualize_sample(
                    preview_dir, 
                    successful_samples, 
                    input_amp_crop,      # 输入振幅
                    output_amp_crop,     # WGS重建振幅
                    output_phase_crop,   # WGS重建相位
                    errors,              # Loss 曲线
                    (start_idx_h, end_idx_h, start_idx_w, end_idx_w)
                )

            successful_samples += 1
            pbar.set_description(f"Saved: {successful_samples}")

        except Exception as e:
            print(f"Error: {e}")
            continue

    print(f"Done. Saved to {dataset_path}")
    print(f"Preview images saved to {preview_dir}")

if __name__ == "__main__":
    # 生成 1000 个样本
    generate_training_samples(10, "./ustc/Others/ai原子阵列重排/gs_training_data/gs_training_data_test_06")