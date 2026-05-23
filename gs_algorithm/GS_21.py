# -*- coding: utf-8 -*-
"""
GS_23_Scalar_Vis.py (完全体：标量自适应 + 论文逻辑 + 可视化监控)

功能集大成：
1. WGS内核: ScalarAdaptiveWGS (解决亮度不均，保证光斑完美高斯形)。
2. 论文逻辑: High Res (1024) 模拟 -> 降采样 -> Low Res (256) 物理场。
3. 数据集: Input (提取相位+缩放坐标), Label (中心裁剪+FFT)。
4. 监控: 自动画 Loss 曲线、对比图，实时查看 WGS 是否收敛。
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import os

# 设置显卡
cp.cuda.Device(0).use()
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 核心算法: 标量自适应 WGS
# ==========================================
class ScalarAdaptiveWGS:
    def __init__(self, slm_size, traps_positions, trap_std=2.0, iterations=50):
        self.xp = cp
        self.slm_size = slm_size
        self.iterations = iterations
        self.trap_std = trap_std
        
        # 核心：保存光镊坐标 (用于测量)
        self.traps_positions = traps_positions
        self.num_traps = len(traps_positions)
        
        # 核心：标量权重数组 (初始全为 1.0)
        self.scalar_weights = self.xp.ones(self.num_traps, dtype=self.xp.float32)
        
        # 预计算坐标网格
        self.y_grid, self.x_grid = self.xp.indices(slm_size)
        
        # 初始相位随机
        self.phase = self.xp.random.rand(*slm_size) * 2 * self.xp.pi
        self.slm_amplitude = self.xp.ones(slm_size)
        self.errors = []
        
        # 初始目标场
        self.current_target_amp = self.rebuild_target_amplitude()

    def rebuild_target_amplitude(self):
        """根据 scalar_weights 动态重绘目标振幅"""
        target = self.xp.zeros(self.slm_size, dtype=self.xp.float32)
        
        for i, (tx, ty) in enumerate(self.traps_positions):
            w = self.scalar_weights[i]
            # 为了速度，只在光镊附近计算
            # 这是一个简单的 bounding box 优化
            y_min, y_max = max(0, ty - 20), min(self.slm_size[0], ty + 20)
            x_min, x_max = max(0, tx - 20), min(self.slm_size[1], tx + 20)
            
            # 局部网格
            # 注意: 这里的切片操作在 GPU 上可能不如全图操作快，取决于光镊数量。
            # 考虑到 Python 循环开销，如果光镊少 (<100)，全图 mask 也可以。
            # 为了稳健，这里还是用全图 mask 方式，但限制计算区域
            
            dist_sq = (self.x_grid - tx)**2 + (self.y_grid - ty)**2
            mask = dist_sq < (self.trap_std * 4)**2 
            
            gaussian = w * self.xp.exp(-dist_sq[mask] / (2 * self.trap_std**2))
            target[mask] = self.xp.maximum(target[mask], gaussian)
            
        return target

    def forward(self, phase):
        slm_field = self.slm_amplitude * self.xp.exp(1j * phase)
        return self.xp.fft.fftshift(self.xp.fft.fft2(self.xp.fft.ifftshift(slm_field)))

    def backward(self, focal_field):
        slm_field = self.xp.fft.fftshift(self.xp.fft.ifft2(self.xp.fft.ifftshift(focal_field)))
        return self.xp.angle(slm_field)

    def update_scalar_weights(self, focal_abs):
        """测量亮度并更新权重"""
        measured_vals = self.xp.zeros(self.num_traps, dtype=self.xp.float32)
        
        for i, (tx, ty) in enumerate(self.traps_positions):
            tx = int(min(max(0, tx), self.slm_size[1]-1))
            ty = int(min(max(0, ty), self.slm_size[0]-1))
            measured_vals[i] = focal_abs[ty, tx]
            
        mean_val = self.xp.mean(measured_vals)
        if mean_val < 1e-6: return
        
        factors = mean_val / (measured_vals + 1e-6)
        alpha = 0.5 
        
        self.scalar_weights *= (1.0 - alpha + alpha * factors)
        # 限制权重范围
        self.scalar_weights = self.xp.clip(self.scalar_weights, 0.1, 1000.0)

    def run(self):
        # 这里的 iterator 用于显示进度
        # 我们把 leave=False 去掉或者设为 True，方便看 log
        for i in range(self.iterations):
            focal_field = self.forward(self.phase)
            focal_abs = self.xp.abs(focal_field)
            focal_phase = self.xp.angle(focal_field)
            
            # 自适应更新 (每5步)
            if i > 5 and i % 5 == 0:
                self.update_scalar_weights(focal_abs)
                self.current_target_amp = self.rebuild_target_amplitude()
            
            # 约束应用 (光镊区强制替换，背景区保留 Current)
            is_trap = self.current_target_amp > 1e-3
            new_amp = focal_abs.copy()
            new_amp[is_trap] = self.current_target_amp[is_trap]
            
            # 反向传播
            new_field = new_amp * self.xp.exp(1j * focal_phase)
            self.phase = self.backward(new_field)
            
            # 记录误差 (Target 与 Current 在光镊区的差异)
            # 这里记录 "不均匀度" 可能更有意义，但为了 Loss 曲线，我们记录 MSE
            diff = (new_amp - focal_abs)**2
            error = self.xp.sum(diff)
            self.errors.append(float(error))

        return cp.asnumpy(self.phase), self.errors

    def get_focal_field(self, phase):
        return self.forward(self.xp.asarray(phase))

# ==========================================
# 2. 辅助函数 (坐标变换与裁剪)
# ==========================================
def generate_trap_coords(slm_size, margin=100, min_traps=5, max_traps=30):
    num_traps = np.random.randint(min_traps, max_traps + 1)
    traps_positions = []
    for _ in range(num_traps):
        while True:
            x = np.random.randint(margin, slm_size[0]-margin)
            y = np.random.randint(margin, slm_size[1]-margin)
            if all((x-tx)**2 + (y-ty)**2 > 400 for tx, ty in traps_positions):
                traps_positions.append((x, y))
                break
    return traps_positions

def extract_phases_high_res(focal_field_high, traps_positions):
    focal_phase = cp.asnumpy(cp.angle(focal_field_high))
    return [focal_phase[y, x] for x, y in traps_positions]

def create_low_res_inputs(low_res_size, traps_positions_high, extracted_phases, scale_factor=4, trap_std=1.0):
    h, w = low_res_size
    input_amp = np.zeros((h, w), dtype=np.float32)
    input_phase = np.zeros((h, w), dtype=np.float32)
    y_idx, x_idx = np.indices((h, w))
    
    for (high_x, high_y), phi in zip(traps_positions_high, extracted_phases):
        low_x = int(high_x / scale_factor)
        low_y = int(high_y / scale_factor)
        low_x = np.clip(low_x, 0, w-1)
        low_y = np.clip(low_y, 0, h-1)
        
        dist_sq = (x_idx - low_x)**2 + (y_idx - low_y)**2
        gaussian = np.exp(-dist_sq / (2 * trap_std**2))
        input_amp = np.maximum(input_amp, gaussian)
        
        mask = dist_sq < (trap_std * 2)**2
        input_phase[mask] = phi
        
    return input_amp, input_phase

def generate_label_from_crop(hologram_high, crop_size=(256, 256)):
    H, W = hologram_high.shape
    ch, cw = crop_size
    start_y, start_x = (H - ch) // 2, (W - cw) // 2
    
    hologram_crop = hologram_high[start_y:start_y+ch, start_x:start_x+cw]
    slm_field_crop = cp.exp(1j * cp.asarray(hologram_crop))
    
    return cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(slm_field_crop)))

# ==========================================
# 3. 可视化函数 (你想要回来的那个！)
# ==========================================
def visualize_check(save_dir, idx, input_amp, input_phase, label_amp, label_phase, errors):
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)
    
    # 1. Input Amp
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(input_amp, cmap='hot', vmin=0, vmax=1)
    ax1.set_title(f'Input Amp (Low Res)')
    
    # 2. Input Phase
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(input_phase, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    ax2.set_title('Input Phase (Extracted)')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # 3. Label Amp (Reconstruction)
    ax3 = fig.add_subplot(gs[1, 0])
    # 动态归一化显示，防止全黑
    vmax_val = np.max(label_amp) if np.max(label_amp) > 0 else 1.0
    im3 = ax3.imshow(label_amp, cmap='hot', vmin=0, vmax=vmax_val)
    ax3.set_title(f'Label Amp (Phys)\nMax: {vmax_val:.4f}')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # 4. Label Phase
    ax4 = fig.add_subplot(gs[1, 1])
    im4 = ax4.imshow(label_phase, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    ax4.set_title('Label Phase')
    
    # 5. WGS Convergence Curve (Error)
    ax5 = fig.add_subplot(gs[:, 2]) # 占右边一整列
    ax5.plot(errors, 'b-', linewidth=1.5)
    ax5.set_title('WGS Convergence (MSE)')
    ax5.set_xlabel('Iterations')
    ax5.set_ylabel('Error')
    ax5.set_yscale('log')
    ax5.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'check_{idx:05d}.png'))
    plt.close()

# ==========================================
# 4. 主流程
# ==========================================
def generate_dataset_final(num_samples, output_dir):
    high_res = (1024, 1024)
    low_res = (256, 256)
    
    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(output_dir, 'training_data_final.h5')
    preview_dir = os.path.join(output_dir, 'previews')
    
    if os.path.exists(h5_path): os.remove(h5_path)
    
    pbar = tqdm(range(num_samples))
    successful = 0
    
    for i in pbar:
        try:
            # 1. 生成坐标
            traps_pos_high = generate_trap_coords(high_res)
            
            # 2. 运行 Scalar Adaptive WGS (100次迭代)
            wgs = ScalarAdaptiveWGS(high_res, traps_pos_high, trap_std=2.0, iterations=1000)
            final_phase_high, errors = wgs.run() 
            
            # 3. 提取特征
            focal_field_high = wgs.get_focal_field(final_phase_high)
            extracted_phases = extract_phases_high_res(focal_field_high, traps_pos_high)
            
            # 4. 生成 Input (256x256)
            input_amp, input_phase = create_low_res_inputs(
                low_res, traps_pos_high, extracted_phases, scale_factor=4
            )
            
            # 5. 生成 Label (256x256)
            focal_field_label = generate_label_from_crop(final_phase_high, crop_size=low_res)
            label_amp = cp.asnumpy(cp.abs(focal_field_label))
            label_phase = cp.asnumpy(cp.angle(focal_field_label))
            
            # 6. 归一化 (为了训练稳定，Label 归一化到 0-1)
            norm = np.max(label_amp) + 1e-10
            label_amp_norm = label_amp / norm
            
            # 7. 保存数据
            with h5py.File(h5_path, 'a') as f:
                grp = f.create_group(f'sample_{successful:05d}')
                grp.create_dataset('input_amplitude', data=input_amp.astype(np.float32), compression='gzip')
                grp.create_dataset('input_phase', data=input_phase.astype(np.float32), compression='gzip')
                grp.create_dataset('output_amplitude', data=label_amp_norm.astype(np.float32), compression='gzip')
                grp.create_dataset('output_phase', data=label_phase.astype(np.float32), compression='gzip')
                grp.create_dataset('num_traps', data=np.array([len(traps_pos_high)]))
                
            # 8. === 可视化监控 (每10张保存一次) ===
            if successful % 10 == 0:
                visualize_check(
                    preview_dir, successful,
                    input_amp, input_phase, 
                    label_amp_norm, label_phase, 
                    errors
                )
                
            successful += 1
            pbar.set_description(f"Saved: {successful}")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    print("Done. Check 'previews' folder.")

if __name__ == "__main__":
    # 生成 1000 个样本
    generate_dataset_final(10, "./ustc/Others/ai原子阵列重排/gs_training_data/gs_training_data_test_14")