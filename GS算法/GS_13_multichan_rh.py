# -*- coding: utf-8 -*-
"""
GS_11_traindata_rh.py (重写版)

此脚本用于生成训练AI模型的数据集，以计算相位全息图。
与原版的主要区别在于，本脚本生成的数据格式与论文中的方法论对齐，
为每个样本创建包含四个部分的数据：

1.  输入 - 通道1 (input_amplitude): 目标光镊位置的振幅分布图。
2.  输入 - 通道2 (input_phase): 为每个目标光镊设定的初始目标相位图。
3.  输出 - 通道1 (output_amplitude): 经过WGS算法优化后，在焦平面重建的振幅图。
4.  输出 - 通道2 (output_phase): 经过WGS算法优化后，得到的最佳相位全息图。

数据集的结构变为：
X_data (输入): shape = (N, 2, H, W)
Y_data (标签): shape = (N, 2, H, W)
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import time
import h5py
import os
from tqdm import tqdm # <--- 新增: 导入tqdm以获得更好的进度条体验

# 设置使用GPU设备
cp.cuda.Device(0).use()

# Matplotlib 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class WeightedGSAlgorithm:
    """
    加权GS算法类 (与原版基本一致)
    使用CuPy在GPU上执行FFT和约束应用，以加速计算。
    """
    def __init__(self, target_amplitude, weights=None, slm_size=(1024, 1024), iterations=50, phase_initial=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.slm_size = slm_size
        self.iterations = iterations

        if use_gpu:
            self.target_amplitude = cp.asarray(target_amplitude)
            self.weights = cp.ones_like(self.target_amplitude) if weights is None else cp.asarray(weights)
            self.phase = cp.random.rand(*slm_size) * 2 * cp.pi if phase_initial is None else cp.asarray(phase_initial)
            self.slm_amplitude = cp.ones(slm_size)
        else: # CPU fallback
            self.target_amplitude = target_amplitude
            self.weights = np.ones_like(target_amplitude) if weights is None else weights
            self.phase = np.random.rand(*slm_size) * 2 * np.pi if phase_initial is None else phase_initial
            self.slm_amplitude = np.ones(slm_size)

        self.errors = []

    def forward_propagation(self, phase):
        slm_field = self.slm_amplitude * cp.exp(1j * phase)
        focal_field = cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(slm_field)))
        return focal_field

    def backward_propagation(self, focal_field):
        slm_field = cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(focal_field)))
        phase = cp.angle(slm_field)
        return phase

    def apply_constraints_focal_plane(self, focal_field):
        current_amplitude = cp.abs(focal_field)
        current_phase = cp.angle(focal_field)
        new_amplitude = self.weights * self.target_amplitude + (1 - self.weights) * current_amplitude
        new_focal_field = new_amplitude * cp.exp(1j * current_phase)
        return new_focal_field

    def calculate_error(self, focal_field):
        current_amplitude = cp.abs(focal_field)
        error = cp.sqrt(cp.sum((current_amplitude - self.target_amplitude)**2))
        self.errors.append(cp.asnumpy(error))
        return error

    def run(self):
        # 使用tqdm来显示迭代进度
        iterator = tqdm(range(self.iterations), desc="WGS Iteration", leave=False)
        for _ in iterator:
            focal_field = self.forward_propagation(self.phase)
            self.calculate_error(focal_field)
            constrained_focal_field = self.apply_constraints_focal_plane(focal_field)
            self.phase = self.backward_propagation(constrained_focal_field)
        return self.get_phase_numpy(), self.errors

    def get_phase_numpy(self):
        return cp.asnumpy(self.phase)


def generate_trap_configuration(slm_size, real_slm_size, min_traps=5, max_traps=50):
    """
    修改: 除了生成光镊位置，还为每个光镊生成一个随机的目标相位。
    返回:
    traps_positions: [(x, y), ...] 坐标列表
    traps_phases: [phi, ...] 相位列表
    """
    num_traps = np.random.randint(min_traps, max_traps + 1)
    traps_positions = []
    traps_phases = [] # <--- 新增: 用于存储每个光镊的目标相位

    center_offset_x = (slm_size[0] - real_slm_size[0]) // 2
    center_offset_y = (slm_size[1] - real_slm_size[1]) // 2

    for _ in range(num_traps):
        x = center_offset_x + np.random.randint(0, real_slm_size[0])
        y = center_offset_y + np.random.randint(0, real_slm_size[1])
        traps_positions.append((x, y))

        # <--- 新增: 为每个光镊生成一个[-pi, pi]的随机相位
        phase = np.random.uniform(-np.pi, np.pi)
        traps_phases.append(phase)

    return traps_positions, traps_phases


def create_target_amplitude(size, traps_positions, trap_std=2.0):
    """创建目标振幅分布 (作为AI的输入通道1)"""
    amplitude = np.zeros(size, dtype=np.float32)
    for pos in traps_positions:
        # 使用更高效的方式生成高斯斑
        y, x = np.indices(size)
        gaussian = np.exp(-((x - pos[0])**2 + (y - pos[1])**2) / (2 * trap_std**2))
        amplitude = np.maximum(amplitude, gaussian) # 使用maximum避免高斯叠加超过1

    # 归一化是可选的，因为后续模型可能需要未归一化的精确位置信息
    # if np.max(amplitude) > 0:
    #     amplitude /= np.max(amplitude)
    return amplitude


def create_input_phase_map(size, traps_positions, traps_phases):
    """
    新增函数: 创建目标相位图 (作为AI的输入通道2)
    这是一个稀疏图，只在光镊位置有相位值，其他地方为0。
    """
    phase_map = np.zeros(size, dtype=np.float32)
    # 将坐标转换为整数索引
    positions_idx = np.array(traps_positions, dtype=int)
    # 直接使用高级索引赋值，效率更高
    phase_map[positions_idx[:, 1], positions_idx[:, 0]] = traps_phases
    return phase_map


def create_weights(size, traps_positions, weight_value=0.98, background_weight=0.98):
    """创建权重矩阵 (逻辑微调，背景权重降低)"""
    weights = np.full(size, background_weight, dtype=np.float32)
    for pos in traps_positions:
        y, x = np.indices(size)
        mask = np.sqrt((x - pos[0])**2 + (y - pos[1])**2) < 5
        weights[mask] = weight_value
    return weights


def save_training_data(dataset_path, data_dict, index):
    """保存训练数据到HDF5文件 (函数本身无需修改)"""
    with h5py.File(dataset_path, 'a') as f:
        group = f.create_group(f'sample_{index:05d}') # 使用5位数索引
        for key, value in data_dict.items():
            group.create_dataset(key, data=value, compression='gzip')


def visualize_results(input_amplitude, input_phase, output_amplitude, output_phase, errors, num_traps, fidelity):
    """
    修改: 可视化所有关键数据，包括双通道输入和双通道输出
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Sample - {num_traps} traps, Fidelity: {fidelity:.4f}', fontsize=16)

    im1 = axes[0, 0].imshow(input_amplitude, cmap='hot')
    axes[0, 0].set_title('Input Amplitude')
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(input_phase, cmap='hsv')
    axes[0, 1].set_title('Input Phase')
    plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[0, 2].imshow(output_phase, cmap='hsv')
    axes[0, 2].set_title('Output Phase')
    plt.colorbar(im3, ax=axes[0, 2])

    im4 = axes[1, 0].imshow(output_amplitude, cmap='hot')
    axes[1, 0].set_title('Output Amplitude')
    plt.colorbar(im4, ax=axes[1, 0])

    # 为了清晰显示稀疏的相位输入，可以放大中心区域
    center_y, center_x = input_phase.shape[0]//2, input_phase.shape[1]//2
    axes[1, 1].imshow(input_phase, cmap='hsv')
    axes[1, 1].set_title('Input Phase')
    axes[1, 1].set_xlim(center_x - 50, center_x + 50)
    axes[1, 1].set_ylim(center_y + 50, center_y - 50)


    axes[1, 2].plot(errors)
    axes[1, 2].set_title('WGS Error')
    axes[1, 2].set_xlabel('Iterations')
    axes[1, 2].set_ylabel('Error')
    axes[1, 2].grid(True)
    axes[1, 2].set_yscale('log')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def generate_training_samples(num_samples, output_dir, slm_size=(1024, 1024), real_slm_size=(256, 256), iterations=200, min_traps=5, max_traps=50):
    """
    主函数 - 生成多组训练数据 (逻辑已根据新数据格式重构)
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset_path = os.path.join(output_dir, 'training_data_multichannel.h5')

    if os.path.exists(dataset_path):
        print(f"警告: 数据集文件 {dataset_path} 已存在，将被覆盖。")
        os.remove(dataset_path)

    start_idx_h = (slm_size[0] - real_slm_size[0]) // 2
    end_idx_h = start_idx_h + real_slm_size[0]
    start_idx_w = (slm_size[1] - real_slm_size[1]) // 2
    end_idx_w = start_idx_w + real_slm_size[1]

    successful_samples = 0
    main_pbar = tqdm(range(num_samples), desc="生成总进度")

    for sample_idx in main_pbar:
        try:
            # 1. <--- 修改: 同时生成位置和相位
            traps_positions, traps_phases = generate_trap_configuration(slm_size, real_slm_size, min_traps, max_traps)
            num_traps = len(traps_positions)
            main_pbar.set_postfix_str(f"当前光镊数: {num_traps}")

            # 2. <--- 修改: 创建双通道输入
            input_amplitude_full = create_target_amplitude(slm_size, traps_positions, trap_std=2.5)
            input_phase_full = create_input_phase_map(slm_size, traps_positions, traps_phases)
            weights_full = create_weights(slm_size, traps_positions, weight_value=0.98, background_weight=0.98)

            # 3. 运行GS算法 (输入仍为振幅)
            wgs = WeightedGSAlgorithm(input_amplitude_full, weights_full, slm_size, iterations, use_gpu=True)
            output_phase_full, errors = wgs.run()

            # 4. 获取重建振幅 (作为输出标签)
            focal_field = wgs.forward_propagation(cp.asarray(output_phase_full))
            output_amplitude_full = cp.asnumpy(cp.abs(focal_field))
            # 归一化重建振幅，使其与输入振幅的范围大致匹配
            if np.max(output_amplitude_full) > 0:
                output_amplitude_full /= np.max(output_amplitude_full)

            # 5. 裁剪所有数据到 real_slm_size
            input_amp_crop = input_amplitude_full[start_idx_h:end_idx_h, start_idx_w:end_idx_w]
            input_phase_crop = input_phase_full[start_idx_h:end_idx_h, start_idx_w:end_idx_w]
            output_amp_crop = output_amplitude_full[start_idx_h:end_idx_h, start_idx_w:end_idx_w]
            output_phase_crop = output_phase_full[start_idx_h:end_idx_h, start_idx_w:end_idx_w]

            # 6. 计算保真度作为质量控制
            overlap = np.sum(input_amp_crop * output_amp_crop)
            norm1 = np.sqrt(np.sum(input_amp_crop**2))
            norm2 = np.sqrt(np.sum(output_amp_crop**2))
            fidelity = overlap / (norm1 * norm2) if (norm1 * norm2) > 0 else 0

            # 7. 只保存高质量样本
            if fidelity > 0.90: # 可以适当提高保真度门槛
                # <--- 修改: 准备包含四个部分的数据字典
                sample_data = {
                    'input_amplitude': input_amp_crop.astype(np.float32),
                    'input_phase': input_phase_crop.astype(np.float32),
                    'output_amplitude': output_amp_crop.astype(np.float32),
                    'output_phase': output_phase_crop.astype(np.float32),
                    'fidelity': np.array([fidelity], dtype=np.float32),
                    'num_traps': np.array([num_traps], dtype=np.int32)
                }
                save_training_data(dataset_path, sample_data, successful_samples)
                
                # 每20个成功样本保存一张预览图
                if successful_samples % 20 == 0:
                    fig = visualize_results(input_amp_crop, input_phase_crop, output_amp_crop, output_phase_crop, errors, num_traps, fidelity)
                    plt.savefig(os.path.join(output_dir, f'preview_{successful_samples:05d}.png'))
                    plt.close(fig)

                successful_samples += 1

        except Exception as e:
            print(f"\n生成样本 {sample_idx + 1} 时出错: {e}")
            continue

    print(f"\n{'='*50}")
    print("数据生成完成!")
    print(f"成功保存样本: {successful_samples}/{num_samples}")
    print(f"数据保存至: {dataset_path}")

    with h5py.File(dataset_path, 'a') as f:
        f.attrs.update({
            'description': 'Multi-channel training data for hologram generation.',
            'num_samples': successful_samples,
            'slm_size': slm_size,
            'real_slm_size': real_slm_size,
            'iterations_per_sample': iterations,
            'creation_date': time.strftime("%Y-%m-%d %H:%M:%S")
        })

if __name__ == "__main__":
    NUM_SAMPLES = 1000      # 想要生成的总样本数
    ITERATIONS = 20000        # 每个样本的WGS迭代次数 (200-500通常足够)
    OUTPUT_DIR = "/home/rh/rh-apps/chatfront20250728/gs_traindata_multichan/gs_traindata_multichan_01"  # 新的输出目录
    MIN_TRAPS = 10
    MAX_TRAPS = 60

    generate_training_samples(
        num_samples=NUM_SAMPLES,
        output_dir=OUTPUT_DIR,
        slm_size=(1024, 1024),
        real_slm_size=(256, 256),
        iterations=ITERATIONS,
        min_traps=MIN_TRAPS,
        max_traps=MAX_TRAPS
    )