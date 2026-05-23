# -*- coding: utf-8 -*-
"""
GS_25_True_Paper.py (硬核论文复刻版)

配置:
- High Res: 8192 x 8192 (WGS 运行环境)
- Low Res:  1024 x 1024 (物理 SLM 尺寸 / AI 输入输出)

逻辑:
1. WGS 在 8192x8192 极其细腻的网格上寻找最优相位。
2. Input: 提取 8192 的相位值 -> 坐标缩放 1/8 -> 在 1024 图上重绘。
3. Label: 截取 8192 全息图的中心 1024 区域 -> FFT -> 模拟真实孔径截断后的光场。

注意: 显存占用较大，请确保没有其他程序占用 GPU。
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import os
import gc

# 设置显卡
cp.cuda.Device(0).use()
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 核心算法: 标量自适应 WGS (支持 8K 分辨率)
# ==========================================
class ScalarAdaptiveWGS_8K:
    def __init__(self, slm_size, traps_positions, trap_std=4.0, iterations=30):
        # 8K 分辨率下，trap_std 需要相应变大，否则在 8192 像素里太小了
        # 论文里 8192 对应全孔径，光斑会比较细腻
        self.xp = cp
        self.slm_size = slm_size
        self.iterations = iterations
        self.trap_std = trap_std
        self.traps_positions = traps_positions
        self.num_traps = len(traps_positions)
        
        # 标量权重
        self.scalar_weights = self.xp.ones(self.num_traps, dtype=self.xp.float32)
        
        # 预生成坐标网格 (注意：8192x8192 的网格占显存，我们动态生成或切片)
        # self.y_grid, self.x_grid = self.xp.indices(slm_size) # 这会消耗 512MB*2，太大了
        # 我们在 rebuild 时只生成局部网格
        
        # 初始相位 (Complex64 以省显存)
        self.phase = (self.xp.random.rand(*slm_size) * 2 * self.xp.pi).astype(self.xp.float32)
        
        # 初始目标场
        self.current_target_amp = self.rebuild_target_amplitude()
        self.errors = []

    def rebuild_target_amplitude(self):
        """局部更新法重绘目标，节省显存"""
        target = self.xp.zeros(self.slm_size, dtype=self.xp.float32)
        
        # 遍历光镊，只在局部画高斯
        for i, (tx, ty) in enumerate(self.traps_positions):
            w = self.scalar_weights[i]
            # 8K 下，光斑范围设大一点，比如 +/- 40 像素
            range_px = int(self.trap_std * 5)
            y_min, y_max = max(0, int(ty) - range_px), min(self.slm_size[0], int(ty) + range_px)
            x_min, x_max = max(0, int(tx) - range_px), min(self.slm_size[1], int(tx) + range_px)
            
            if y_min >= y_max or x_min >= x_max: continue

            # 生成局部网格
            y_sub, x_sub = self.xp.ogrid[y_min:y_max, x_min:x_max]
            dist_sq = (x_sub - ty)**2 + (y_sub - tx)**2 # 注意 x,y 对应 col, row
            
            mask = dist_sq < (self.trap_std * 4)**2
            gaussian = w * self.xp.exp(-dist_sq / (2 * self.trap_std**2))
            
            # 叠加
            target_sub = target[y_min:y_max, x_min:x_max]
            temp = self.xp.zeros_like(target_sub)
            temp[mask] = gaussian[mask]
            target[y_min:y_max, x_min:x_max] = self.xp.maximum(target_sub, temp)
            
        return target

    def forward(self, phase):
        # 8K FFT
        # slm_amplitude 默认为 1，直接算 exp(1j*phase)
        slm_field = self.xp.exp(1j * phase)
        return self.xp.fft.fftshift(self.xp.fft.fft2(self.xp.fft.ifftshift(slm_field)))

    def backward(self, focal_field):
        slm_field = self.xp.fft.fftshift(self.xp.fft.ifft2(self.xp.fft.ifftshift(focal_field)))
        return self.xp.angle(slm_field)

    def update_scalar_weights(self, focal_abs):
        measured_vals = self.xp.zeros(self.num_traps, dtype=self.xp.float32)
        for i, (tx, ty) in enumerate(self.traps_positions):
            tx, ty = int(tx), int(ty)
            if 0 <= tx < self.slm_size[1] and 0 <= ty < self.slm_size[0]:
                measured_vals[i] = focal_abs[ty, tx]
        
        mean_val = self.xp.mean(measured_vals)
        if mean_val < 1e-6: return
        
        factors = mean_val / (measured_vals + 1e-6)
        self.scalar_weights *= (1.0 - 0.5 + 0.5 * factors)
        # 8K 下能量分散，权重可能需要更大，上限调高
        self.scalar_weights = self.xp.clip(self.scalar_weights, 0.1, 5000.0)

    def run(self):
        # 减少迭代次数以节省时间，8K 收敛较慢但物理更准
        for i in range(self.iterations):
            focal_field = self.forward(self.phase)
            focal_abs = self.xp.abs(focal_field)
            focal_phase = self.xp.angle(focal_field)
            
            if i > 3 and i % 3 == 0: # 8K 比较慢，减少更新频率
                self.update_scalar_weights(focal_abs)
                self.current_target_amp = self.rebuild_target_amplitude()
            
            is_trap = self.current_target_amp > 1e-3
            new_amp = focal_abs.copy() # 这里会申请 512MB
            new_amp[is_trap] = self.current_target_amp[is_trap]
            
            # 反向
            new_field = new_amp * self.xp.exp(1j * focal_phase)
            self.phase = self.backward(new_field)
            
            # 清理中间变量
            del new_amp, new_field, focal_field
            # 记录误差 (采样几个点即可，不用算全图MSE，太慢)
            self.errors.append(float(self.xp.mean(self.scalar_weights))) # 记录权重变化代替误差

        # 最后一次前向计算用于输出
        final_field = self.forward(self.phase)
        return cp.asnumpy(self.phase), final_field

# ==========================================
# 2. 辅助函数 (8K -> 1K 映射)
# ==========================================
def generate_trap_coords(slm_size, margin=400, min_traps=5, max_traps=30):
    # 在 8192 范围内生成坐标
    num_traps = np.random.randint(min_traps, max_traps + 1)
    traps_positions = []
    for _ in range(num_traps):
        while True:
            x = np.random.randint(margin, slm_size[0]-margin)
            y = np.random.randint(margin, slm_size[1]-margin)
            # 距离判断 (8K下距离要大一点)
            if all((x-tx)**2 + (y-ty)**2 > 1600 for tx, ty in traps_positions):
                traps_positions.append((x, y))
                break
    return traps_positions

def extract_phases_8k(focal_field_8k, traps_positions):
    """从 8192 光场中提取相位"""
    # 为了省内存，不要把整个 8192x8192 转到 CPU
    # 既然我们知道坐标，直接在 GPU 上索引
    extracted_phases = []
    # focal_field_8k 是 GPU array
    phase_map = cp.angle(focal_field_8k)
    
    for tx, ty in traps_positions:
        # 注意：traps_positions 是 (x, y)，array索引是 [y, x]
        p = phase_map[int(ty), int(tx)]
        extracted_phases.append(float(p))
    
    del phase_map # 释放
    return extracted_phases

def create_input_1k(size_1k, traps_positions_8k, extracted_phases, scale_factor=8, trap_std=2.0):
    """
    Input: 8K 坐标 -> 除以 8 -> 在 1K 上重绘
    """
    h, w = size_1k
    input_amp = np.zeros((h, w), dtype=np.float32)
    input_phase = np.zeros((h, w), dtype=np.float32)
    y_idx, x_idx = np.indices((h, w))
    
    for (high_x, high_y), phi in zip(traps_positions_8k, extracted_phases):
        # 坐标映射
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

def generate_label_crop_1k(slm_phase_8k):
    """
    Label: 8K SLM -> 截取中心 1K -> FFT -> Label
    """
    # slm_phase_8k 是 numpy 数组 (CPU)
    H, W = slm_phase_8k.shape
    target_size = 1024
    
    start_y = (H - target_size) // 2
    start_x = (W - target_size) // 2
    
    # 1. 裁剪全息图 (中心 1024x1024)
    # 这一步模拟真实的有限孔径 SLM
    slm_crop_phase = slm_phase_8k[start_y:start_y+target_size, start_x:start_x+target_size]
    
    # 2. 转回 GPU 做 FFT
    slm_field_crop = cp.exp(1j * cp.asarray(slm_crop_phase))
    focal_field_label = cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(slm_field_crop)))
    
    # 3. 获取结果
    label_amp = cp.asnumpy(cp.abs(focal_field_label))
    label_phase = cp.asnumpy(cp.angle(focal_field_label))
    
    del slm_field_crop, focal_field_label # 及时释放
    return label_amp, label_phase

# ==========================================
# 3. 可视化
# ==========================================
def visualize_true_paper(save_dir, idx, in_amp, in_phi, lab_amp, lab_phi):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0,0].imshow(in_amp, cmap='hot', vmin=0, vmax=1)
    axes[0,0].set_title('Input Amp (Redrawn from 8K coords)')
    
    axes[0,1].imshow(in_phi, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[0,1].set_title('Input Phase (Extracted from 8K WGS)')
    
    # Label 的亮度可能会因为裁剪而变低，这是物理现象
    # 我们归一化显示来看看光斑形态
    norm_val = np.max(lab_amp) + 1e-10
    axes[1,0].imshow(lab_amp, cmap='hot', vmin=0, vmax=norm_val)
    axes[1,0].set_title(f'Label Amp (FFT of Cropped SLM)\nPhys Max: {norm_val:.2e}')
    
    axes[1,1].imshow(lab_phi, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[1,1].set_title('Label Phase')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'check_paper_{idx:05d}.png'))
    plt.close()

# ==========================================
# 4. 主流程
# ==========================================
def generate_dataset_true_paper(num_samples, output_dir):
    res_8k = (8192, 8192)
    res_1k = (1024, 1024)
    
    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(output_dir, 'training_data_8k_to_1k.h5')
    preview_dir = os.path.join(output_dir, 'previews_8k')
    
    if os.path.exists(h5_path): os.remove(h5_path)
    
    pbar = tqdm(range(num_samples))
    successful = 0
    
    for i in pbar:
        try:
            # 显存清理：每一步都清理，防止 8K 堆积
            cp.get_default_memory_pool().free_all_blocks()
            
            # 1. 生成 8K 坐标
            traps_pos_8k = generate_trap_coords(res_8k)
            
            # 2. 运行 8K WGS
            # std 设为 8.0，因为 8192 像素里，光斑需要占一定份量才能在 FFT 后被"看见"
            wgs = ScalarAdaptiveWGS_8K(res_8k, traps_pos_8k, trap_std=8.0, iterations=20)
            final_phase_8k, focal_field_8k = wgs.run()
            
            # 3. 生成 Input (1024)
            # 提取相位
            extracted_phases = extract_phases_8k(focal_field_8k, traps_pos_8k)
            # 释放 8K 光场显存 (很重要!)
            del focal_field_8k 
            cp.get_default_memory_pool().free_all_blocks()
            
            # 重绘
            input_amp, input_phase = create_input_1k(
                res_1k, traps_pos_8k, extracted_phases, scale_factor=8, trap_std=2.0
            )
            
            # 4. 生成 Label (1024)
            # 裁剪 -> FFT
            label_amp, label_phase = generate_label_crop_1k(final_phase_8k)
            
            # 5. 归一化 Label
            norm = np.max(label_amp) + 1e-10
            label_amp_norm = label_amp / norm
            
            # 6. 保存
            with h5py.File(h5_path, 'a') as f:
                grp = f.create_group(f'sample_{successful:05d}')
                grp.create_dataset('input_amplitude', data=input_amp.astype(np.float32), compression='gzip')
                grp.create_dataset('input_phase', data=input_phase.astype(np.float32), compression='gzip')
                grp.create_dataset('output_amplitude', data=label_amp_norm.astype(np.float32), compression='gzip')
                grp.create_dataset('output_phase', data=label_phase.astype(np.float32), compression='gzip')
                grp.create_dataset('num_traps', data=np.array([len(traps_pos_8k)]))
            
            if successful % 5 == 0:
                visualize_true_paper(preview_dir, successful, input_amp, input_phase, label_amp_norm, label_phase)
                
            successful += 1
            pbar.set_description(f"Saved: {successful} (VRAM OK)")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            # 发生错误时也要尝试清理显存
            cp.get_default_memory_pool().free_all_blocks()
            continue
            
    print("Done.")

if __name__ == "__main__":
    # 试跑 100 个，看看显存顶不顶得住
    generate_dataset_true_paper(100, "./ustc/Others/ai原子阵列重排/gs_training_data/gs_training_data_8k_paper")