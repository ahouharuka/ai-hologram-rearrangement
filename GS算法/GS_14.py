# -*- coding: utf-8 -*-
"""
GS_13_multichan_rh_fixed.py (最终修正版)

修正内容:
1. WGS算法逻辑: 增加了【归一化】步骤，解决背景全白/不收敛的问题。
2. 代码兼容性: 修复了类方法中混用 cp 和 self.xp 的 bug，现在可以自由切换 CPU/GPU。
3. 调试功能: debug_wgs 现在包含完整的防报错机制。
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import os

# Matplotlib 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class WeightedGSAlgorithm:
    """
    加权GS算法 (修正版)
    """
    def __init__(self, target_amplitude, weights=None, slm_size=(1024, 1024), iterations=50, phase_initial=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.slm_size = slm_size
        self.iterations = iterations

        # === 1. 定义计算后端 xp (Numpy 或 Cupy) ===
        if use_gpu:
            self.xp = cp
        else:
            self.xp = np

        # 初始化数据，全部使用 self.xp
        self.target_amplitude = self.xp.asarray(target_amplitude)
        self.weights = self.xp.ones_like(self.target_amplitude) if weights is None else self.xp.asarray(weights)
        
        if phase_initial is None:
            self.phase = self.xp.random.rand(*slm_size) * 2 * self.xp.pi
        else:
            self.phase = self.xp.asarray(phase_initial)
            
        self.slm_amplitude = self.xp.ones(slm_size)
        self.errors = []

    def forward_propagation(self, phase):
        # 修正: 使用 self.xp 而不是 cp
        slm_field = self.slm_amplitude * self.xp.exp(1j * phase)
        focal_field = self.xp.fft.fftshift(self.xp.fft.fft2(self.xp.fft.ifftshift(slm_field)))
        return focal_field

    def backward_propagation(self, focal_field):
        # 修正: 使用 self.xp 而不是 cp
        slm_field = self.xp.fft.fftshift(self.xp.fft.ifft2(self.xp.fft.ifftshift(focal_field)))
        phase = self.xp.angle(slm_field)
        return phase

    def apply_constraints_focal_plane(self, focal_field):
        # 修正: 使用 self.xp 而不是 cp
        current_amplitude = self.xp.abs(focal_field)
        current_phase = self.xp.angle(focal_field)
        
        # === 核心修正: 归一化 (解决背景全白问题) ===
        # 将当前的物理振幅归一化到 [0, 1]，使其能与 target_amplitude 有效混合
        max_val = self.xp.max(current_amplitude)
        if max_val > 1e-10:
            current_amplitude_norm = current_amplitude / max_val
        else:
            current_amplitude_norm = current_amplitude

        # 使用归一化后的振幅进行混合
        new_amplitude = self.weights * self.target_amplitude + (1 - self.weights) * current_amplitude_norm
        
        # 组合回新的复数场
        new_focal_field = new_amplitude * self.xp.exp(1j * current_phase)
        return new_focal_field

    def calculate_error(self, focal_field):
        # 修正: 使用 self.xp 而不是 cp
        current_amplitude = self.xp.abs(focal_f ield)
        
        # 为了计算误差，这里也最好做一个归一化对比，或者直接计算原始误差
        # 这里保持原样即可，主要影响 Loss 曲线的数值大小
        error = self.xp.sqrt(self.xp.sum((current_amplitude - self.target_amplitude)**2))
        
        # 转换回 CPU 存入列表
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
        
        # 返回结果 (统一转为 Numpy 方便后续处理)
        if self.use_gpu:
            return cp.asnumpy(self.phase), self.errors
        else:
            return self.phase, self.errors

# === 辅助函数保持不变 ===
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

def create_weights(size, traps_positions, weight_value=0.98): 
    weights = np.full(size, 0.1, dtype=np.float32)
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

def generate_training_samples(num_samples, output_dir, slm_size=(1024, 1024), real_slm_size=(256, 256), iterations=50):
    os.makedirs(output_dir, exist_ok=True)
    dataset_path = os.path.join(output_dir, 'training_data_multichannel_fixed.h5')

    if os.path.exists(dataset_path):
        os.remove(dataset_path)

    start_idx_h = (slm_size[0] - real_slm_size[0]) // 2
    end_idx_h = start_idx_h + real_slm_size[0]
    start_idx_w = (slm_size[1] - real_slm_size[1]) // 2
    end_idx_w = start_idx_w + real_slm_size[1]

    successful_samples = 0
    pbar = tqdm(range(num_samples))

    for sample_idx in pbar:
        try:
            traps_pos, traps_phi = generate_trap_configuration(slm_size, real_slm_size)
            
            input_amp_full = create_target_amplitude(slm_size, traps_pos, trap_std=2.5)
            weights_full = create_weights(slm_size, traps_pos)
            
            # 运行 WGS (GPU)
            wgs = WeightedGSAlgorithm(input_amp_full, weights_full, slm_size, iterations, use_gpu=True)
            final_slm_phase, errors = wgs.run()

            # 计算 Label (需要将 final_slm_phase 转回 GPU 进行正向传播)
            phase_gpu = cp.asarray(final_slm_phase)
            focal_field_complex = wgs.forward_propagation(phase_gpu)
            
            label_amp_full = cp.asnumpy(cp.abs(focal_field_complex))
            label_phase_full = cp.asnumpy(cp.angle(focal_field_complex))
            
            input_amp_data = input_amp_full
            input_phase_data = create_input_phase_map(slm_size, traps_pos, traps_phi, trap_std=2.5)

            def crop(arr): return arr[start_idx_h:end_idx_h, start_idx_w:end_idx_w]
            
            # 对 Label 振幅也进行归一化，确保存入的数据在 [0,1]
            norm_factor = np.max(label_amp_full) + 1e-8
            
            sample_data = {
                'input_amplitude': crop(input_amp_data).astype(np.float32),
                'input_phase': crop(input_phase_data).astype(np.float32),
                'output_amplitude': (crop(label_amp_full) / norm_factor).astype(np.float32),
                'output_phase': crop(label_phase_full).astype(np.float32),
                'num_traps': np.array([len(traps_pos)])
            }
            
            save_training_data(dataset_path, sample_data, successful_samples)
            successful_samples += 1
            pbar.set_description(f"Saved: {successful_samples}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"Done. Saved to {dataset_path}")

def debug_wgs():
    """
    调试函数 (增强版)
    """
    print("正在使用 GPU 运行 WGS 调试 (最终修正版)...")
    slm_size = (1024, 1024)
    
    # 1. 生成数据
    traps_pos = [(512, 512)] 
    input_amp = create_target_amplitude(slm_size, traps_pos, trap_std=2.0)
    weights = create_weights(slm_size, traps_pos)
    
    # 2. 运行 WGS
    wgs = WeightedGSAlgorithm(input_amp, weights, slm_size, iterations=50, use_gpu=True)
    final_phase, errors = wgs.run()
    
    # 3. 验证结果
    # 确保类型匹配
    if wgs.use_gpu:
        phase_input = cp.asarray(final_phase)
    else:
        phase_input = final_phase
        
    focal_field = wgs.forward_propagation(phase_input)
    
    # 强制拉回 CPU 绘图
    if wgs.use_gpu:
        recon_amp = cp.asnumpy(cp.abs(focal_field))
    else:
        recon_amp = np.abs(focal_field)
    
    # 归一化显示，以便观察
    recon_amp_norm = recon_amp / (np.max(recon_amp) + 1e-10)

    # 4. 画图
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Input Goal (Ideal)")
    plt.imshow(input_amp[450:574, 450:574], cmap='hot')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.title("WGS Reconstruction (Actual)")
    # 这里应该能看到黑底亮斑了
    plt.imshow(recon_amp_norm[450:574, 450:574], cmap='hot', vmin=0, vmax=1)
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.title("Hologram Phase")
    plt.imshow(final_phase if isinstance(final_phase, np.ndarray) else cp.asnumpy(final_phase), cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # debug_wgs()
    generate_training_samples(1000, "./ustc/Others/ai原子阵列重排/gs_training_data/gs_training_data_03")