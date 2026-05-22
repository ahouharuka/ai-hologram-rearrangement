# -*- coding: utf-8 -*-
"""
GS_20_Paper_LowRes.py (论文逻辑-硬件降级版)

分辨率映射:
- High Res (模拟层): 1024 x 1024 (对应论文 8192)
- Low Res (物理层): 256 x 256 (对应论文 1024)

流程:
1. 在 1024 层生成坐标，运行无相位约束 WGS。
2. Input生成: 提取 1024 层的相位 -> 坐标缩放 / 4 -> 在 256 层重绘 A_input, Phi_input。
3. Label生成: 截取 1024 全息图中心 256 -> FFT -> 得到 256 层的 A_label, Phi_label。
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
# 1. 纯振幅约束 WGS (运行在 1024 High Res)
# ==========================================
# ==========================================
# 1. 纯振幅约束 + 自适应权重 WGS (修复版)
# ==========================================
class PhaseFreeWGS:
    def __init__(self, target_amplitude, weights=None, slm_size=(1024, 1024), iterations=50):
        self.xp = cp
        self.slm_size = slm_size
        self.iterations = iterations
        self.target_amplitude = self.xp.asarray(target_amplitude)
        
        # 初始权重
        self.weights = self.xp.ones_like(self.target_amplitude) if weights is None else self.xp.asarray(weights)
        
        # 生成一个掩模，标记光镊的位置 (用于检测亮度)
        # 只要 target > 0.5 (高斯峰值附近) 就认为是光镊核心区域
        self.trap_mask = self.target_amplitude > 0.5
        
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

    def update_adaptive_weights(self, current_amp):
        """
        【关键修复】自适应权重更新
        动态调整每个光镊的权重，强制均匀性。
        """
        # 1. 提取所有光镊区域的当前亮度
        # 注意：这里我们简化处理，直接用 mask 区域的平均值来代表各个光镊
        # 更严谨的做法是传入光镊坐标，但用 mask 也可以凑合，因为光镊是分散的
        
        # 为了精确控制，我们需要知道哪些像素属于同一个光镊。
        # 但在 GPU 上做连通域分析太慢。
        # 这里采用一个近似策略：直接对 mask 区域内的每个像素进行像素级调整。
        
        vals = current_amp[self.trap_mask]
        if vals.size == 0: return

        mean_val = self.xp.mean(vals)
        
        # 2. 计算修正因子
        # 如果某像素比平均值暗 (current < mean)，则 factor > 1，权重增加
        # 如果某像素比平均值亮 (current > mean)，则 factor < 1，权重减小
        # 加上 1e-6 防止除零
        correction = mean_val / (vals + 1e-6)
        
        # 3. 软更新 (避免震荡)
        alpha = 0.5 # 更新步长
        
        # 更新权重
        self.weights[self.trap_mask] *= (1 - alpha + alpha * correction)
        
        # 限制权重上限，防止爆炸 (比如最大不超过 100)
        self.weights = self.xp.clip(self.weights, 0, 100.0)

    def run(self):
        # 进度条
        iterator = tqdm(range(self.iterations), desc="WGS (High Res)", leave=False)
        
        for i in iterator:
            focal_field = self.forward(self.phase)
            current_amp = self.xp.abs(focal_field)
            current_phase = self.xp.angle(focal_field)
            
            # === 自适应权重更新 ===
            # 每 10 次迭代更新一次权重，让光斑亮度趋于一致
            if i % 10 == 0 and i < self.iterations - 10: # 最后几步不更新，确保存储
                self.update_adaptive_weights(current_amp)

            # 动态归一化 (匹配能量)
            max_val = self.xp.max(current_amp)
            if max_val < 1e-10: max_val = 1.0
            
            curr_norm = current_amp / max_val
            
            # 混合逻辑 (Phase-Free)
            # 在光镊位置: 强力拉向 Target
            # 在背景位置: 权重极小，跟随 Current
            new_amp = self.weights * self.target_amplitude + (1 - self.weights > 0) * (1 - self.weights) * curr_norm
            
            # 这里原本的混合公式在权重很大(>1)时会出问题：(1-weights) 会变成负数
            # 让我们换一个更稳定的“信号替换”逻辑：
            # 如果是光镊区域 (mask)，直接用 weights 调节后的 target
            # 如果是背景，保留 current
            
            # === 修正混合公式 ===
            # 将 weights 理解为 "Target的增益"
            # 我们构建一个 Ideal Amp，它的形状是 Target，但通过 weights 进行了局部增强
            # 简单的替换策略：
            # New = Target (在光镊处) + Current (在背景处)
            # 但光镊处的 Target 需要乘上 adaptive weights 才能把暗的提起来
            
            # 使用最经典的 WGS 加权策略:
            # 仅在 trap_mask 区域应用约束
            
            signal = self.weights * self.target_amplitude # 光镊区目标
            background = curr_norm * (1.0 - self.trap_mask) # 背景区保留
            
            # 组合 (注意：self.weights 在背景区应该设为0或很小，这里我们在 create_target 时设了 0.1)
            # 但上面的 update_adaptive 可能会把背景权重搞乱，所以要小心
            
            # 更稳妥的写法：
            # 1. 光镊区：用 weights * target 替换
            # 2. 背景区：完全保留 current (自由演化)
            
            new_amp_combined = self.xp.zeros_like(current_amp)
            new_amp_combined[self.trap_mask] = signal[self.trap_mask] # 强制替换
            new_amp_combined[~self.trap_mask] = curr_norm[~self.trap_mask] # 背景自由
            
            # 反归一化
            new_amp_scaled = new_amp_combined * max_val
            
            # 组合新场
            new_field = new_amp_scaled * self.xp.exp(1j * current_phase)
            self.phase = self.backward(new_field)
            
            # 记录误差
            error = self.xp.sum((curr_norm - self.target_amplitude)**2)
            self.errors.append(float(error))
            
        return cp.asnumpy(self.phase), self.errors
    
    def get_focal_field(self, phase):
        return self.forward(self.xp.asarray(phase))

# ==========================================
# 2. 核心处理函数 (坐标变换与裁剪)
# ==========================================
def create_high_res_target(size, traps_positions, trap_std=2.0):
    """在 1024 层创建目标振幅和权重"""
    amplitude = np.zeros(size, dtype=np.float32)
    weights = np.ones(size, dtype=np.float32) * 0.1
    y_idx, x_idx = np.indices(size)
    
    for pos in traps_positions:
        # 振幅
        gaussian = np.exp(-((x_idx - pos[0])**2 + (y_idx - pos[1])**2) / (2 * trap_std**2))
        amplitude = np.maximum(amplitude, gaussian)
        # 权重
        mask = ((x_idx - pos[0])**2 + (y_idx - pos[1])**2) < 25 
        weights[mask] = 20.0
        
    return amplitude, weights

def extract_phases_high_res(focal_field_high, traps_positions):
    """从 1024 层的场中提取光镊中心的相位值"""
    focal_phase = cp.asnumpy(cp.angle(focal_field_high))
    extracted_phases = []
    for pos in traps_positions:
        x, y = pos
        extracted_phases.append(focal_phase[y, x])
    return extracted_phases

def create_low_res_inputs(low_res_size, traps_positions_high, extracted_phases, scale_factor=4, trap_std=1.0):
    """
    【Input 生成逻辑】
    1. 接收 1024 层的坐标和相位。
    2. 坐标缩小 4 倍 (只取整数)。
    3. 在 256 层重绘 Input Amp 和 Input Phase。
    """
    h, w = low_res_size
    input_amp = np.zeros((h, w), dtype=np.float32)
    input_phase = np.zeros((h, w), dtype=np.float32)
    y_idx, x_idx = np.indices((h, w))
    
    for (high_x, high_y), phi in zip(traps_positions_high, extracted_phases):
        # === 核心操作 1: 坐标缩小 ===
        low_x = int(high_x / scale_factor)
        low_y = int(high_y / scale_factor)
        
        # 边界保护
        low_x = np.clip(low_x, 0, w-1)
        low_y = np.clip(low_y, 0, h-1)
        
        # === 核心操作 2: 重绘 (Redraw) ===
        # 注意: trap_std 也应该相应缩小，比如从 2.0 变成 0.5 或者 1.0，视 AI 需求而定
        # 这里为了让 CNN 容易读，我们保持 std=1.0 左右，不要太小
        dist_sq = (x_idx - low_x)**2 + (y_idx - low_y)**2
        
        # 画振幅
        gaussian = np.exp(-dist_sq / (2 * trap_std**2))
        input_amp = np.maximum(input_amp, gaussian)
        
        # 画相位 (半径 2 内)
        mask = dist_sq < (trap_std * 2)**2
        input_phase[mask] = phi
        
    return input_amp, input_phase

def generate_label_from_crop(hologram_high, crop_size=(256, 256)):
    """
    【Label 生成逻辑】
    1. 截取 1024 全息图的中心 256 区域。
    2. FFT 得到物理场。
    """
    H, W = hologram_high.shape
    ch, cw = crop_size
    
    # === 核心操作 3: 中心裁剪 ===
    start_y = (H - ch) // 2
    start_x = (W - cw) // 2
    hologram_crop = hologram_high[start_y:start_y+ch, start_x:start_x+cw]
    
    # 这里的 hologram_crop 依然是相位全息图 (complex exp(1j*phi))
    # 在 WGS 类里 phase 是 float，所以这里要转成复数场
    # 注意: WGS output 出来的是 phase (float)
    slm_field_crop = cp.exp(1j * cp.asarray(hologram_crop))
    
    # === 核心操作 4: FFT 得到 Label ===
    focal_field_label = cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(slm_field_crop)))
    
    return focal_field_label

# ==========================================
# 3. 辅助: 随机坐标生成
# ==========================================
def generate_trap_coords(slm_size, margin=100, min_traps=5, max_traps=30):
    num_traps = np.random.randint(min_traps, max_traps + 1)
    traps_positions = []
    
    for _ in range(num_traps):
        while True:
            # 在 1024 范围内生成，留出足够边距防止裁剪后丢失
            # 我们的裁剪是中心裁剪，所以光镊必须集中在中心 256 区域对应的放大区域内？
            # 这是一个坑！
            # 如果 Label 是截取全息图中心，全息图中心对应的焦平面视野是全视野。
            # 不对！截取全息图会导致焦平面光斑变大（NA变小），但视野范围（FOV）是由像素间距决定的。
            # 如果像素间距不变，FOV 不变。
            # 但是，为了安全，我们最好把光镊生成在中心区域，或者全图都可以？
            # 论文中 "8192 resolution" 对应全孔径。截取 1024 对应孔径光阑。
            # 光镊位置还是在那个物理平面上。
            # 简单起见，我们在全图生成，但要避开最边缘。
            
            x = np.random.randint(margin, slm_size[0]-margin)
            y = np.random.randint(margin, slm_size[1]-margin)
            
            if all((x-tx)**2 + (y-ty)**2 > 400 for tx, ty in traps_positions):
                traps_positions.append((x, y))
                break
    return traps_positions

# ==========================================
# 4. 可视化检查
# ==========================================
def visualize_check(save_dir, idx, in_amp, in_phi, lab_amp, lab_phi):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes[0,0].imshow(in_amp, cmap='hot')
    axes[0,0].set_title(f'Input Amp (256x256)\nScaled from 1024')
    
    axes[0,1].imshow(in_phi, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[0,1].set_title('Input Phase (256x256)\nRedrawn with Extracted Value')
    
    axes[1,0].imshow(lab_amp, cmap='hot')
    axes[1,0].set_title('Label Amp (256x256)\nFFT of Cropped Hologram')
    
    axes[1,1].imshow(lab_phi, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[1,1].set_title('Label Phase (256x256)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'check_{idx:03d}.png'))
    plt.close()

# ==========================================
# 5. 主程序
# ==========================================
def generate_paper_data(num_samples, output_dir):
    high_res = (1024, 1024)
    low_res = (256, 256)
    
    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(output_dir, 'training_data_paper_repro.h5')
    if os.path.exists(h5_path): os.remove(h5_path)
    
    pbar = tqdm(range(num_samples))
    successful = 0
    
    for i in pbar:
        try:
            # 1. 生成 1024 层坐标
            # 注意：光镊不一定要在中心。为了利用全视场，我们在整个 1024 范围内生成。
            traps_pos_high = generate_trap_coords(high_res)
            
            # 2. 运行 High Res WGS
            target_amp, weights = create_high_res_target(high_res, traps_pos_high)
            wgs = PhaseFreeWGS(target_amp, weights, high_res, iterations=1000) # 100次足够收敛
            final_phase_high, _ = wgs.run() # 这是 1024x1024 的全息图相位
            
            # 3. 提取特征 (Feature Extraction)
            focal_field_high = wgs.get_focal_field(final_phase_high)
            extracted_phases = extract_phases_high_res(focal_field_high, traps_pos_high)
            
            # 4. 生成 Input (256x256)
            input_amp, input_phase = create_low_res_inputs(
                low_res, traps_pos_high, extracted_phases, scale_factor=4
            )
            
            # 5. 生成 Label (256x256)
            # 截取中心 -> FFT
            focal_field_label = generate_label_from_crop(final_phase_high, crop_size=low_res)
            
            label_amp = cp.asnumpy(cp.abs(focal_field_label))
            label_phase = cp.asnumpy(cp.angle(focal_field_label))
            
            # 6. 归一化 Label Amp
            norm = np.max(label_amp) + 1e-10
            label_amp_norm = label_amp / norm
            
            # 7. 保存
            with h5py.File(h5_path, 'a') as f:
                grp = f.create_group(f'sample_{successful:05d}')
                grp.create_dataset('input_amplitude', data=input_amp.astype(np.float32), compression='gzip')
                grp.create_dataset('input_phase', data=input_phase.astype(np.float32), compression='gzip')
                grp.create_dataset('output_amplitude', data=label_amp_norm.astype(np.float32), compression='gzip')
                grp.create_dataset('output_phase', data=label_phase.astype(np.float32), compression='gzip')
                grp.create_dataset('num_traps', data=np.array([len(traps_pos_high)]))
            
            # 可视化监控
            if successful % 10 == 0:
                visualize_check(os.path.join(output_dir, 'previews'), successful, 
                                input_amp, input_phase, label_amp_norm, label_phase)
                
            successful += 1
            pbar.set_description(f"Saved: {successful}")
            
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    generate_paper_data(10, "./ustc/Others/ai原子阵列重排/gs_training_data/gs_training_data_test_12")