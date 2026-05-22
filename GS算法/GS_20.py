# -*- coding: utf-8 -*-
"""
GS_22_Scalar_Adaptive.py (标量权重最终修正版)

核心逻辑:
1. 维护一个长度为 N 的标量权重数组 (对应 N 个光镊)。
2. 每次迭代测量 N 个点的峰值亮度。
3. 根据 (平均亮度 / 只有亮度) 更新标量权重。
4. 动态重绘 Target Amplitude，保证高斯形状不被破坏。
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

class ScalarAdaptiveWGS:
    def __init__(self, slm_size, traps_positions, trap_std=2.0, iterations=50):
        self.xp = cp
        self.slm_size = slm_size
        self.iterations = iterations
        self.trap_std = trap_std
        
        # 核心：保存光镊坐标 (用于测量)
        # 转换为 GPU 数组方便计算距离，或者保留 CPU 列表
        self.traps_positions = traps_positions
        self.num_traps = len(traps_positions)
        
        # 核心：标量权重数组 (初始全为 1.0)
        self.scalar_weights = self.xp.ones(self.num_traps, dtype=self.xp.float32)
        
        # 预计算坐标网格 (用于快速重绘 Target)
        self.y_grid, self.x_grid = self.xp.indices(slm_size)
        
        # 初始相位随机
        self.phase = self.xp.random.rand(*slm_size) * 2 * self.xp.pi
        self.slm_amplitude = self.xp.ones(slm_size)
        self.errors = []
        
        # 初始目标场 (基于权重 1.0)
        self.current_target_amp = self.rebuild_target_amplitude()

    def rebuild_target_amplitude(self):
        """
        根据当前的 scalar_weights 动态重绘目标振幅图。
        Target = Sum( w_k * Gaussian_k )
        """
        target = self.xp.zeros(self.slm_size, dtype=self.xp.float32)
        
        # 这里为了速度，可以只重绘局部，但为了准确先重绘全图
        # 如果光镊很多，这里可以用核函数加速，但Python循环几十次也很快
        for i, (tx, ty) in enumerate(self.traps_positions):
            # 获取该光镊当前的权重
            w = self.scalar_weights[i]
            
            # 生成高斯
            # 只有当权重 > 0 时才画
            dist_sq = (self.x_grid - tx)**2 + (self.y_grid - ty)**2
            # 这里的 mask 稍微大一点，保证高斯完整
            mask = dist_sq < (self.trap_std * 4)**2 
            
            gaussian = w * self.xp.exp(-dist_sq[mask] / (2 * self.trap_std**2))
            
            # 叠加到 Target 上 (使用 maximum 或 add 均可，光镊分开时 maximum 更好)
            # 注意 Cupy 的索引赋值
            target[mask] = self.xp.maximum(target[mask], gaussian)
            
        return target

    def forward(self, phase):
        slm_field = self.slm_amplitude * self.xp.exp(1j * phase)
        return self.xp.fft.fftshift(self.xp.fft.fft2(self.xp.fft.ifftshift(slm_field)))

    def backward(self, focal_field):
        slm_field = self.xp.fft.fftshift(self.xp.fft.ifft2(self.xp.fft.ifftshift(focal_field)))
        return self.xp.angle(slm_field)

    def update_scalar_weights(self, focal_abs):
        """
        测量每个点的实际亮度，更新标量权重。
        """
        # 1. 测量亮度
        measured_vals = self.xp.zeros(self.num_traps, dtype=self.xp.float32)
        
        # 直接读取坐标点的像素值 (近似 Peak 值)
        # 如果想更准，可以取坐标周围 3x3 的平均值
        for i, (tx, ty) in enumerate(self.traps_positions):
            # 边界保护
            tx = int(min(max(0, tx), self.slm_size[1]-1))
            ty = int(min(max(0, ty), self.slm_size[0]-1))
            measured_vals[i] = focal_abs[ty, tx]
            
        # 2. 计算均匀性指标
        mean_val = self.xp.mean(measured_vals)
        if mean_val < 1e-6: return # 防止全黑时除零
        
        # 3. 计算修正因子
        # 越暗的点 (val < mean)，需要越大的因子
        # 限制因子范围，防止震荡 (0.5 ~ 1.5)
        factors = mean_val / (measured_vals + 1e-6)
        # 引入阻尼 (Damping)，不要一次调到位
        alpha = 0.5 
        
        # 4. 更新权重
        # w_new = w_old * (1 - alpha + alpha * factor)
        self.scalar_weights *= (1.0 - alpha + alpha * factors)
        
        # 归一化权重 (防止整体数值无限膨胀)
        # 让权重的平均值保持在 1.0 (或者其他常数)
        # 这一步对于防止"全黑"或"过曝"至关重要
        # weight_mean = self.xp.mean(self.scalar_weights)
        # self.scalar_weights /= weight_mean 
        # *修正*: 在Phase-Free模式下，为了竞争能量，不需要强制归一化，
        # 但为了数值稳定，可以限制最大值
        self.scalar_weights = self.xp.clip(self.scalar_weights, 0.1, 1000.0)

    def run(self):
        iterator = tqdm(range(self.iterations), desc="Adaptive WGS", leave=False)
        
        for i in iterator:
            # 1. 前向传播
            focal_field = self.forward(self.phase)
            focal_abs = self.xp.abs(focal_field)
            focal_phase = self.xp.angle(focal_field)
            
            # 2. 自适应权重更新 (每 5 步一次，前 10 步不更)
            if i > 5 and i % 5 == 0:
                self.update_scalar_weights(focal_abs)
                # 重要：权重变了，目标场也要重绘！
                self.current_target_amp = self.rebuild_target_amplitude()
            
            # 3. 约束应用 (Phase-Free)
            # 策略：振幅替换。
            # 在光镊区域：强制替换为 Target (由 scalar weights 支撑的 Target)
            # 在背景区域：保留 Current (或者压暗，取决于是否想要黑背景)
            
            # 这里我们使用"软替换"策略，保留相位自由度
            # 动态归一化 Current 以匹配 Target 的能量级别，避免 Target 显得太暗或太亮
            # 简单的做法：Target 是由 Weight 决定的，Weight 很大，所以 Target 很大。
            # 我们直接用 Target 替换 Current 的振幅。
            
            # 区分光镊区和背景区
            # 如果 current_target_amp > 0，说明是光镊区
            is_trap = self.current_target_amp > 1e-3
            
            new_amp = focal_abs.copy()
            # 强制替换光镊区振幅
            new_amp[is_trap] = self.current_target_amp[is_trap]
            
            # 背景区：可以选择让它自由演化 (new_amp 保持 focal_abs)
            # 或者稍微压暗一点 (new_amp *= 0.9) 以提高对比度
            # 论文中通常允许背景自由演化以获得更好的光镊效率
            
            # 4. 反向传播
            new_field = new_amp * self.xp.exp(1j * focal_phase)
            self.phase = self.backward(new_field)
            
            # 监控均匀性 (Uniformity)
            if i % 10 == 0:
                vals = []
                for tx, ty in self.traps_positions:
                    vals.append(focal_abs[ty, tx])
                vals = self.xp.array(vals)
                uni = 1 - (self.xp.max(vals) - self.xp.min(vals)) / (self.xp.max(vals) + self.xp.min(vals))
                iterator.set_postfix(Uniformity=f"{float(uni):.3f}")

        return cp.asnumpy(self.phase), self.errors

    def get_focal_field(self, phase):
        return self.forward(self.xp.asarray(phase))

# ==========================================
# 2. 辅助与主程序
# ==========================================

def generate_paper_data_v2(num_samples, output_dir):
    """
    更新后的主生成函数，使用 ScalarAdaptiveWGS
    """
    high_res = (1024, 1024)
    low_res = (256, 256)
    
    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(output_dir, 'training_data_adaptive.h5')
    if os.path.exists(h5_path): os.remove(h5_path)
    
    pbar = tqdm(range(num_samples))
    successful = 0
    
    # 导入之前的辅助函数 (坐标生成等)
    # 这里假设你保留了 GS_20_Paper_LowRes.py 里的辅助函数
    # create_low_res_inputs, extract_phases_high_res, generate_label_from_crop
    # 以及 generate_trap_coords
    
    from GS_19 import generate_trap_coords, extract_phases_high_res, create_low_res_inputs, generate_label_from_crop

    for i in pbar:
        try:
            # 1. 生成坐标
            traps_pos_high = generate_trap_coords(high_res)
            
            # 2. 运行 Scalar Adaptive WGS
            # 注意：不再需要 create_high_res_target，类内部会自动处理
            wgs = ScalarAdaptiveWGS(high_res, traps_pos_high, trap_std=2.0, iterations=1000)
            final_phase_high, _ = wgs.run() 
            
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
            
            # 6. 归一化
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
                
            successful += 1
            
        except Exception as e:
            print(f"Error: {e}")
            continue
            
    print("Done.")

if __name__ == "__main__":
    generate_paper_data_v2(10, "./ustc/Others/ai原子阵列重排/gs_training_data/gs_training_data_test_13")