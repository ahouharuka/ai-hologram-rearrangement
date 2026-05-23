# -*- coding: utf-8 -*-
"""
GS_19_PhaseFree.py (PRL 论文复刻版)

核心逻辑变更:
1. WGS 阶段: 【完全不限制相位】。只约束振幅，让 WGS 自由演化出物理上最优的相位分布。
2. 数据构造: 
   - Input: 坐标位置 + 【事后提取的 WGS 相位】
   - Label: WGS 生成的完整复振幅场
3. 目的: AI 学习的是 "Input Coord + Input Phase -> Physical Field" 的物理映射。
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import os

# 设置显卡
cp.cuda.Device(0).use()

# Matplotlib 中文设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 纯振幅约束 WGS (Phase-Free WGS)
# ==========================================
class PhaseFreeWGS:
    def __init__(self, target_amplitude, weights=None, slm_size=(1024, 1024), iterations=50):
        self.xp = cp
        self.slm_size = slm_size
        self.iterations = iterations

        self.target_amplitude = self.xp.asarray(target_amplitude)
        
        # 权重: 依然需要，用于强迫光镊处亮，背景黑
        self.weights = self.xp.ones_like(self.target_amplitude) if weights is None else self.xp.asarray(weights)
        
        # 初始相位随机
        self.phase = self.xp.random.rand(*slm_size) * 2 * self.xp.pi
        self.slm_amplitude = self.xp.ones(slm_size)
        self.errors = []

    def forward(self, phase):
        slm_field = self.slm_amplitude * self.xp.exp(1j * phase)
        return self.xp.fft.fftshift(self.xp.fft.fft2(self.xp.fft.ifftshift(slm_field)))

    def backward(self, focal_field):
        slm_field = self.xp.fft.fftshift(self.xp.fft.ifft2(self.xp.fft.ifftshift(focal_field)))
        return self.xp.angle(slm_field)

    def run(self):
        iterator = tqdm(range(self.iterations), desc="Phase-Free WGS", leave=False)
        for i in iterator:
            focal_field = self.forward(self.phase)
            current_amp = self.xp.abs(focal_field)
            current_phase = self.xp.angle(focal_field)
            
            # --- 核心逻辑: 只替换振幅，保留相位 ---
            # 归一化 current 以匹配 target 量级 (防止能量耗散导致的全黑)
            # 我们动态调整 target 的强度来匹配 current 的平均能量
            # 或者更简单：直接混合形状
            
            max_val = self.xp.max(current_amp)
            if max_val < 1e-10: max_val = 1.0
            
            # 归一化后的 current
            curr_norm = current_amp / max_val
            
            # 混合振幅: New_Amp = Weight * Target + (1-Weight) * Current_Norm
            # 注意: Target 已经是 0-1 的理想高斯
            new_amp = self.weights * self.target_amplitude + (1 - self.weights) * curr_norm
            
            # 反归一化回去 (保持能量守恒)
            new_amp_scaled = new_amp * max_val
            
            # 组合新的场: 使用新的振幅 + 【原本的自然相位】
            new_field = new_amp_scaled * self.xp.exp(1j * current_phase)
            
            self.phase = self.backward(new_field)
            
            # 记录误差
            error = self.xp.sum(self.xp.abs(curr_norm - self.target_amplitude)**2)
            self.errors.append(float(error))
            
        return cp.asnumpy(self.phase), self.errors

    def get_focal_field(self, phase):
        return self.forward(self.xp.asarray(phase))

# ==========================================
# 2. 辅助函数
# ==========================================
def generate_trap_coords(slm_size, real_slm_size, min_traps=5, max_traps=40):
    """只生成坐标，不生成相位"""
    num_traps = np.random.randint(min_traps, max_traps + 1)
    traps_positions = []
    
    center_offset_x = (slm_size[0] - real_slm_size[0]) // 2
    center_offset_y = (slm_size[1] - real_slm_size[1]) // 2

    # 简单的防重叠生成
    for _ in range(num_traps):
        while True:
            x = center_offset_x + np.random.randint(20, real_slm_size[0]-20)
            y = center_offset_y + np.random.randint(20, real_slm_size[1]-20)
            # 简单检查距离
            if all((x-tx)**2 + (y-ty)**2 > 100 for tx, ty in traps_positions):
                traps_positions.append((x, y))
                break
                
    return traps_positions

def create_target_amplitude(size, traps_positions, trap_std=2.0):
    amplitude = np.zeros(size, dtype=np.float32)
    y_idx, x_idx = np.indices(size)
    for pos in traps_positions:
        gaussian = np.exp(-((x_idx - pos[0])**2 + (y_idx - pos[1])**2) / (2 * trap_std**2))
        amplitude = np.maximum(amplitude, gaussian)
    return amplitude

def create_weights(size, traps_positions, weight_value=20.0): 
    weights = np.ones(size, dtype=np.float32) * 0.1 
    y_idx, x_idx = np.indices(size)
    for pos in traps_positions:
        mask = ((x_idx - pos[0])**2 + (y_idx - pos[1])**2) < 25 
        weights[mask] = weight_value
    return weights

def extract_measured_phases(focal_field, traps_positions, size, trap_std=2.0):
    """
    【关键步骤】
    从 WGS 跑出来的结果中，提取光镊位置的相位。
    这将作为 AI 的 Input Phase。
    """
    focal_phase = cp.angle(focal_field) # GPU array
    phase_map = np.zeros(size, dtype=np.float32) # CPU array
    y_idx, x_idx = np.indices(size)
    
    # 将相位场转回 CPU 以便操作
    focal_phase_cpu = cp.asnumpy(focal_phase)
    
    for pos in traps_positions:
        x, y = pos
        # 获取该点中心的相位值
        measured_phi = focal_phase_cpu[y, x]
        
        # 将这个相位值画在 Input Map 上 (高斯扩散)
        dist_sq = (x_idx - x)**2 + (y_idx - y)**2
        mask = dist_sq < (trap_std * 2)**2 
        phase_map[mask] = measured_phi
        
    return phase_map

def save_training_data(dataset_path, data_dict, index):
    with h5py.File(dataset_path, 'a') as f:
        group = f.create_group(f'sample_{index:05d}')
        for key, value in data_dict.items():
            group.create_dataset(key, data=value, compression='gzip')

def visualize_sample(save_dir, index, input_amp, input_phase, recon_amp, recon_phase):
    """可视化: 检查 Input Phase 是否真的取自 Recon Phase"""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes[0,0].imshow(input_amp, cmap='hot')
    axes[0,0].set_title('Input Amp (Ideal)')
    
    axes[0,1].imshow(input_phase, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[0,1].set_title('Input Phase (Extracted from WGS)')
    
    axes[1,0].imshow(recon_amp, cmap='hot')
    axes[1,0].set_title('Label Amp (WGS Result)')
    
    axes[1,1].imshow(recon_phase, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[1,1].set_title('Label Phase (WGS Result)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'check_{index}.png'))
    plt.close()

# ==========================================
# 3. 主流程
# ==========================================
def generate_training_samples(num_samples, output_dir, slm_size=(1024, 1024), real_slm_size=(256, 256), iterations=2000):
    os.makedirs(output_dir, exist_ok=True)
    dataset_path = os.path.join(output_dir, 'training_data_phasefree.h5')
    preview_dir = os.path.join(output_dir, 'previews')
    
    if os.path.exists(dataset_path): os.remove(dataset_path)

    # 裁剪区域
    start_idx_h = (slm_size[0] - real_slm_size[0]) // 2
    end_idx_h = start_idx_h + real_slm_size[0]
    start_idx_w = (slm_size[1] - real_slm_size[1]) // 2
    end_idx_w = start_idx_w + real_slm_size[1]

    successful_samples = 0
    pbar = tqdm(range(num_samples))

    for sample_idx in pbar:
        try:
            # 1. 生成坐标 (No Phase yet!)
            traps_pos = generate_trap_coords(slm_size, real_slm_size)
            
            # 2. 准备 WGS 输入 (只管振幅)
            target_amp_full = create_target_amplitude(slm_size, traps_pos, trap_std=2.0)
            weights_full = create_weights(slm_size, traps_pos, weight_value=20.0)
            
            # 3. 运行 Phase-Free WGS
            # 让算法自由寻找相位，不加约束
            wgs = PhaseFreeWGS(target_amp_full, weights_full, slm_size, iterations)
            final_slm_phase, errors = wgs.run()

            # 4. 获取 WGS 结果 (Ground Truth)
            focal_field = wgs.get_focal_field(final_slm_phase)
            
            # 5. 【关键】事后提取相位作为 Input
            # 我们去测量 WGS 到底选了什么相位，然后告诉 AI: "这就是你该学的相位"
            input_phase_extracted = extract_measured_phases(focal_field, traps_pos, slm_size)
            
            # 6. 准备数据保存
            label_amp_full = cp.asnumpy(cp.abs(focal_field))
            label_phase_full = cp.asnumpy(cp.angle(focal_field))
            
            # 归一化 Label Amp
            norm = np.max(label_amp_full) + 1e-10
            
            def crop(arr): return arr[start_idx_h:end_idx_h, start_idx_w:end_idx_w]
            
            sample_data = {
                'input_amplitude': crop(target_amp_full).astype(np.float32),
                'input_phase': crop(input_phase_extracted).astype(np.float32), # 提取出的相位
                'output_amplitude': (crop(label_amp_full) / norm).astype(np.float32),
                'output_phase': crop(label_phase_full).astype(np.float32),
                'num_traps': np.array([len(traps_pos)])
            }
            
            save_training_data(dataset_path, sample_data, successful_samples)
            
            # 可视化监控
            if successful_samples % 10 == 0:
                visualize_sample(preview_dir, successful_samples, 
                                 sample_data['input_amplitude'], sample_data['input_phase'],
                                 sample_data['output_amplitude'], sample_data['output_phase'])
            
            successful_samples += 1
            pbar.set_description(f"Saved: {successful_samples}")

        except Exception as e:
            print(f"Error: {e}")
            continue

    print(f"Done. Saved to {dataset_path}")

if __name__ == "__main__":
    generate_training_samples(10, "./ustc/Others/ai原子阵列重排/gs_training_data/gs_training_data_test_07")