"""
CFG Diagnostic Tests
Copy the cells below into your notebook to diagnose the fluidigmc1 outlier issue.
"""

# ============================================================
# Cell 1: Setup and Basic Statistics
# ============================================================
diagnostic_cell_1 = """
# Cell 1: Check batch distribution and basic stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 检查各batch的样本数和基本统计
print("="*60)
print("Batch Distribution in Training Data")
print("="*60)

batch_counts = batch_labels.value_counts()
print(batch_counts)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 样本数
axes[0].bar(range(len(batch_counts)), batch_counts.values)
axes[0].set_xticks(range(len(batch_counts)))
axes[0].set_xticklabels(batch_counts.index, rotation=45, ha='right')
axes[0].set_ylabel('Number of Samples')
axes[0].set_title('Samples per Batch')

# L2范数分布
batch_names = batch_labels.unique()
norms_by_batch = []
for batch in batch_names:
    mask = batch_labels == batch
    norms = np.linalg.norm(latent_adata.X[mask], axis=1)
    norms_by_batch.append(norms)

bp = axes[1].boxplot(norms_by_batch, labels=batch_names, patch_artist=True)
axes[1].set_xticklabels(batch_names, rotation=45, ha='right')
axes[1].set_ylabel('L2 Norm')
axes[1].set_title('L2 Norm Distribution by Batch (Real Data)')

plt.tight_layout()
plt.show()

# 特别关注fluidigmc1
print("\\n" + "="*60)
print("fluidigmc1 Statistics")
print("="*60)
fluidigmc1_mask = batch_labels == 'fluidigmc1'
fluidigmc1_data = latent_adata.X[fluidigmc1_mask]
print(f"样本数: {fluidigmc1_mask.sum()}")
print(f"均值: {fluidigmc1_data.mean():.4f}")
print(f"标准差: {fluidigmc1_data.std():.4f}")
print(f"L2范数均值: {np.linalg.norm(fluidigmc1_data, axis=1).mean():.4f}")
"""

# ============================================================
# Cell 2: Test Model Forward with Different Labels
# ============================================================
diagnostic_cell_2 = """
# Cell 2: 测试模型对不同batch的forward输出

import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(batch_labels)

# 获取模型和设备
model = diffusion.ema_model if diffusion.hparams.use_ema else diffusion.model
model.eval()
device = next(model.parameters()).device

# 创建一个固定的噪声输入和时间步
torch.manual_seed(42)
n_test = 100
x_test = torch.randn(n_test, diffusion.hparams.input_dim, device=device)
t_test = torch.full((n_test,), 500, device=device, dtype=torch.long)  # 中间时间步

# 测试每个batch的条件预测
print("="*60)
print("Model Forward Output Statistics by Batch")
print("="*60)

results = {}
batch_names = le.classes_

# 首先测试无条件预测
with torch.no_grad():
    # 无条件预测
    null_output = model.forward(x_test, t_test, labels=None, cond_drop_prob=1.0)
    results['unconditional'] = {
        'mean': null_output.mean().item(),
        'std': null_output.std().item(),
        'norm': torch.norm(null_output, dim=1).mean().item(),
        'output': null_output.cpu().numpy()
    }
    
    # 每个batch的条件预测
    for batch_name in batch_names:
        batch_idx = le.transform([batch_name])[0]
        batch_labels_tensor = torch.full((n_test,), batch_idx, device=device, dtype=torch.long)
        
        cond_output = model.forward(x_test, t_test, labels=batch_labels_tensor)
        
        # 计算与无条件预测的差异
        diff = cond_output - null_output
        
        results[batch_name] = {
            'mean': cond_output.mean().item(),
            'std': cond_output.std().item(),
            'norm': torch.norm(cond_output, dim=1).mean().item(),
            'diff_norm': torch.norm(diff, dim=1).mean().item(),
            'diff_std': diff.std().item(),
            'output': cond_output.cpu().numpy()
        }

# 打印结果
print(f"\\n{'Batch':<15} {'Mean':>10} {'Std':>10} {'Norm':>10} {'Diff Norm':>12} {'Diff Std':>10}")
print("-" * 70)
print(f"{'unconditional':<15} {results['unconditional']['mean']:>10.4f} {results['unconditional']['std']:>10.4f} {results['unconditional']['norm']:>10.4f} {'N/A':>12} {'N/A':>10}")
for batch_name in batch_names:
    r = results[batch_name]
    print(f"{batch_name:<15} {r['mean']:>10.4f} {r['std']:>10.4f} {r['norm']:>10.4f} {r['diff_norm']:>12.4f} {r['diff_std']:>10.4f}")

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Std comparison
stds = [results[b]['std'] for b in batch_names]
axes[0].bar(range(len(batch_names)), stds)
axes[0].axhline(results['unconditional']['std'], color='red', linestyle='--', label='Unconditional')
axes[0].set_xticks(range(len(batch_names)))
axes[0].set_xticklabels(batch_names, rotation=45, ha='right')
axes[0].set_ylabel('Output Std')
axes[0].set_title('Conditional Output Std by Batch')
axes[0].legend()

# Diff norm
diff_norms = [results[b]['diff_norm'] for b in batch_names]
axes[1].bar(range(len(batch_names)), diff_norms)
axes[1].set_xticks(range(len(batch_names)))
axes[1].set_xticklabels(batch_names, rotation=45, ha='right')
axes[1].set_ylabel('|cond - uncond| Norm')
axes[1].set_title('Difference from Unconditional by Batch')

# Diff std
diff_stds = [results[b]['diff_std'] for b in batch_names]
axes[2].bar(range(len(batch_names)), diff_stds)
axes[2].set_xticks(range(len(batch_names)))
axes[2].set_xticklabels(batch_names, rotation=45, ha='right')
axes[2].set_ylabel('Std of (cond - uncond)')
axes[2].set_title('Std of Difference by Batch')

plt.tight_layout()
plt.show()

# 检查是否有异常batch
print("\\n" + "="*60)
print("Anomaly Detection")
print("="*60)
mean_diff_norm = np.mean(diff_norms)
std_diff_norm = np.std(diff_norms)
for i, batch_name in enumerate(batch_names):
    if diff_norms[i] > mean_diff_norm + 2 * std_diff_norm:
        print(f"⚠️  {batch_name}: diff_norm={diff_norms[i]:.4f} is unusually HIGH!")
    elif diff_norms[i] < mean_diff_norm - 2 * std_diff_norm:
        print(f"⚠️  {batch_name}: diff_norm={diff_norms[i]:.4f} is unusually LOW!")
"""

# ============================================================
# Cell 3: Test CFG Step by Step
# ============================================================
diagnostic_cell_3 = """
# Cell 3: 逐步测试CFG过程，找出问题所在

import torch
import numpy as np
from functools import partial

# 获取模型
model = diffusion.ema_model if diffusion.hparams.use_ema else diffusion.model
model.eval()
device = next(model.parameters()).device

# 创建测试输入
torch.manual_seed(42)
n_test = 100
x_test = torch.randn(n_test, diffusion.hparams.input_dim, device=device)
t_test = torch.full((n_test,), 500, device=device, dtype=torch.long)

# 测试每个batch
le = LabelEncoder()
le.fit(batch_labels)

print("="*60)
print("CFG Step-by-Step Analysis")
print("="*60)

def std_fn(x):
    return torch.std(x, dim=tuple(range(1, x.ndim)), keepdim=True)

rescaled_phi = 0.7
cond_scale = 0.0  # 我们测试的情况

all_results = {}

with torch.no_grad():
    # 无条件预测
    null_logits = model.forward(x_test, t_test, labels=None, cond_drop_prob=1.0)
    null_std = std_fn(null_logits).mean().item()
    
    print(f"\\nUnconditional (null_logits):")
    print(f"  Mean: {null_logits.mean().item():.4f}")
    print(f"  Std:  {null_std:.4f}")
    print(f"  Norm: {torch.norm(null_logits, dim=1).mean().item():.4f}")
    
    for batch_name in le.classes_:
        batch_idx = le.transform([batch_name])[0]
        labels = torch.full((n_test,), batch_idx, device=device, dtype=torch.long)
        
        # Step 1: 条件预测
        logits = model.forward(x_test, t_test, labels=labels)
        logits_std = std_fn(logits).mean().item()
        
        # Step 2: update (cond - uncond)
        update = logits - null_logits
        
        # Step 3: scaled_logits (with cond_scale=0.0)
        scaled_logits = null_logits + update * cond_scale  # = null_logits when cond_scale=0
        scaled_std = std_fn(scaled_logits).mean().item()
        
        # Step 4: rescaling
        std_ratio = std_fn(logits) / (std_fn(scaled_logits) + 1e-8)
        rescaled_logits = scaled_logits * std_ratio
        
        # Step 5: interpolation
        final = rescaled_logits * rescaled_phi + scaled_logits * (1.0 - rescaled_phi)
        
        # 关键：比较final和null_logits的差异
        diff_from_null = final - null_logits
        diff_norm = torch.norm(diff_from_null, dim=1).mean().item()
        
        all_results[batch_name] = {
            'logits_std': logits_std,
            'scaled_std': scaled_std,
            'std_ratio': std_ratio.mean().item(),
            'diff_from_null': diff_norm,
            'final_norm': torch.norm(final, dim=1).mean().item(),
            'final': final.cpu().numpy()
        }
        
        print(f"\\n{batch_name}:")
        print(f"  logits std:       {logits_std:.4f}")
        print(f"  scaled_logits std:{scaled_std:.4f}")
        print(f"  std ratio:        {std_ratio.mean().item():.4f}")
        print(f"  |final - null|:   {diff_norm:.4f}")

# 可视化关键指标
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

batch_names = list(all_results.keys())

# Std ratio (logits_std / scaled_std)
std_ratios = [all_results[b]['std_ratio'] for b in batch_names]
colors = ['red' if r > 1.5 or r < 0.7 else 'blue' for r in std_ratios]
axes[0].bar(range(len(batch_names)), std_ratios, color=colors)
axes[0].axhline(1.0, color='green', linestyle='--', label='Expected (1.0)')
axes[0].set_xticks(range(len(batch_names)))
axes[0].set_xticklabels(batch_names, rotation=45, ha='right')
axes[0].set_ylabel('std(logits) / std(scaled_logits)')
axes[0].set_title('Std Ratio by Batch\\n(Should be ~1.0 for cond_scale=0)')
axes[0].legend()

# Difference from null
diff_from_nulls = [all_results[b]['diff_from_null'] for b in batch_names]
colors = ['red' if d > 0.1 else 'blue' for d in diff_from_nulls]
axes[1].bar(range(len(batch_names)), diff_from_nulls, color=colors)
axes[1].axhline(0.0, color='green', linestyle='--', label='Expected (0.0)')
axes[1].set_xticks(range(len(batch_names)))
axes[1].set_xticklabels(batch_names, rotation=45, ha='right')
axes[1].set_ylabel('|final - null_logits|')
axes[1].set_title('Deviation from Unconditional\\n(Should be ~0 for cond_scale=0)')
axes[1].legend()

# Final norm comparison with unconditional
final_norms = [all_results[b]['final_norm'] for b in batch_names]
null_norm = torch.norm(null_logits, dim=1).mean().item()
axes[2].bar(range(len(batch_names)), final_norms)
axes[2].axhline(null_norm, color='red', linestyle='--', label=f'Unconditional ({null_norm:.2f})')
axes[2].set_xticks(range(len(batch_names)))
axes[2].set_xticklabels(batch_names, rotation=45, ha='right')
axes[2].set_ylabel('Final Output Norm')
axes[2].set_title('Final Output Norm by Batch')
axes[2].legend()

plt.tight_layout()
plt.show()

# 找出问题batch
print("\\n" + "="*60)
print("Problem Identification")
print("="*60)
for batch_name in batch_names:
    r = all_results[batch_name]
    issues = []
    if r['std_ratio'] > 1.5:
        issues.append(f"std_ratio too high ({r['std_ratio']:.2f})")
    if r['std_ratio'] < 0.7:
        issues.append(f"std_ratio too low ({r['std_ratio']:.2f})")
    if r['diff_from_null'] > 0.1:
        issues.append(f"differs from unconditional ({r['diff_from_null']:.4f})")
    
    if issues:
        print(f"⚠️  {batch_name}: {', '.join(issues)}")
"""

# ============================================================
# Cell 4: Deep Dive into fluidigmc1
# ============================================================
diagnostic_cell_4 = """
# Cell 4: 深入分析fluidigmc1

import torch
import numpy as np

# 获取模型
model = diffusion.ema_model if diffusion.hparams.use_ema else diffusion.model
model.eval()
device = next(model.parameters()).device

le = LabelEncoder()
le.fit(batch_labels)

torch.manual_seed(42)
n_test = 1000  # 更多样本

x_test = torch.randn(n_test, diffusion.hparams.input_dim, device=device)

print("="*60)
print("Deep Dive: fluidigmc1 vs Other Batches")
print("="*60)

# 测试不同时间步
timesteps = [0, 100, 300, 500, 700, 900, 999]

def std_fn(x):
    return torch.std(x, dim=tuple(range(1, x.ndim)), keepdim=True)

results_by_t = {t: {} for t in timesteps}

with torch.no_grad():
    for t_val in timesteps:
        t_test = torch.full((n_test,), t_val, device=device, dtype=torch.long)
        
        # 无条件
        null_logits = model.forward(x_test, t_test, labels=None, cond_drop_prob=1.0)
        null_std = std_fn(null_logits)
        
        # fluidigmc1
        fluidigmc1_idx = le.transform(['fluidigmc1'])[0]
        fluidigmc1_labels = torch.full((n_test,), fluidigmc1_idx, device=device, dtype=torch.long)
        fluidigmc1_logits = model.forward(x_test, t_test, labels=fluidigmc1_labels)
        fluidigmc1_std = std_fn(fluidigmc1_logits)
        
        # 随机选一个其他batch (比如celseq2)
        other_idx = le.transform(['celseq2'])[0] if 'celseq2' in le.classes_ else le.transform([le.classes_[0]])[0]
        other_labels = torch.full((n_test,), other_idx, device=device, dtype=torch.long)
        other_logits = model.forward(x_test, t_test, labels=other_labels)
        other_std = std_fn(other_logits)
        
        results_by_t[t_val] = {
            'null_std': null_std.mean().item(),
            'fluidigmc1_std': fluidigmc1_std.mean().item(),
            'other_std': other_std.mean().item(),
            'fluidigmc1_ratio': (fluidigmc1_std / (null_std + 1e-8)).mean().item(),
            'other_ratio': (other_std / (null_std + 1e-8)).mean().item(),
            'fluidigmc1_diff': (fluidigmc1_logits - null_logits).abs().mean().item(),
            'other_diff': (other_logits - null_logits).abs().mean().item(),
        }

# 打印结果
print(f"\\n{'Timestep':>8} {'Null Std':>10} {'Fluid Std':>10} {'Other Std':>10} {'Fluid Ratio':>12} {'Other Ratio':>12}")
print("-" * 75)
for t_val in timesteps:
    r = results_by_t[t_val]
    print(f"{t_val:>8} {r['null_std']:>10.4f} {r['fluidigmc1_std']:>10.4f} {r['other_std']:>10.4f} {r['fluidigmc1_ratio']:>12.4f} {r['other_ratio']:>12.4f}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

t_vals = timesteps
fluidigmc1_ratios = [results_by_t[t]['fluidigmc1_ratio'] for t in t_vals]
other_ratios = [results_by_t[t]['other_ratio'] for t in t_vals]

axes[0].plot(t_vals, fluidigmc1_ratios, 'r-o', label='fluidigmc1')
axes[0].plot(t_vals, other_ratios, 'b-o', label='celseq2 (reference)')
axes[0].axhline(1.0, color='green', linestyle='--', label='Expected (1.0)')
axes[0].set_xlabel('Timestep')
axes[0].set_ylabel('std(cond) / std(uncond)')
axes[0].set_title('Std Ratio Across Timesteps')
axes[0].legend()

fluidigmc1_diffs = [results_by_t[t]['fluidigmc1_diff'] for t in t_vals]
other_diffs = [results_by_t[t]['other_diff'] for t in t_vals]

axes[1].plot(t_vals, fluidigmc1_diffs, 'r-o', label='fluidigmc1')
axes[1].plot(t_vals, other_diffs, 'b-o', label='celseq2 (reference)')
axes[1].set_xlabel('Timestep')
axes[1].set_ylabel('|cond - uncond| Mean')
axes[1].set_title('Difference from Unconditional Across Timesteps')
axes[1].legend()

plt.tight_layout()
plt.show()
"""

# ============================================================
# Cell 5: Check Label Embedding
# ============================================================
diagnostic_cell_5 = """
# Cell 5: 检查Label Embedding

import torch
import numpy as np
import torch.nn.functional as F

model = diffusion.ema_model if diffusion.hparams.use_ema else diffusion.model
model.eval()
device = next(model.parameters()).device

le = LabelEncoder()
le.fit(batch_labels)

print("="*60)
print("Label Embedding Analysis")
print("="*60)

# 获取所有batch的label embedding
label_embeddings = {}
with torch.no_grad():
    # Null embedding
    null_emb = model.null_label_emb.detach().cpu().numpy()
    label_embeddings['null'] = null_emb
    print(f"\\nNull embedding norm: {np.linalg.norm(null_emb):.4f}")
    
    for batch_name in le.classes_:
        batch_idx = le.transform([batch_name])[0]
        labels = torch.tensor([batch_idx], device=device, dtype=torch.long)
        
        # 获取label embedding
        emb = model.label_embedding(labels).detach().cpu().numpy()[0]
        label_embeddings[batch_name] = emb
        
        # 计算与null的距离
        dist_from_null = np.linalg.norm(emb - null_emb)
        cos_sim_with_null = np.dot(emb, null_emb) / (np.linalg.norm(emb) * np.linalg.norm(null_emb) + 1e-8)
        
        print(f"\\n{batch_name}:")
        print(f"  Embedding norm:      {np.linalg.norm(emb):.4f}")
        print(f"  Distance from null:  {dist_from_null:.4f}")
        print(f"  Cosine sim with null:{cos_sim_with_null:.4f}")

# 可视化embedding距离矩阵
all_names = ['null'] + list(le.classes_)
n_batches = len(all_names)
distance_matrix = np.zeros((n_batches, n_batches))

for i, name1 in enumerate(all_names):
    for j, name2 in enumerate(all_names):
        distance_matrix[i, j] = np.linalg.norm(label_embeddings[name1] - label_embeddings[name2])

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(distance_matrix, cmap='viridis')
ax.set_xticks(range(n_batches))
ax.set_yticks(range(n_batches))
ax.set_xticklabels(all_names, rotation=45, ha='right')
ax.set_yticklabels(all_names)
ax.set_title('Label Embedding Distance Matrix')
plt.colorbar(im)

# 标注值
for i in range(n_batches):
    for j in range(n_batches):
        ax.text(j, i, f'{distance_matrix[i, j]:.2f}', ha='center', va='center', fontsize=8)

plt.tight_layout()
plt.show()

# 检查fluidigmc1是否有异常
print("\\n" + "="*60)
print("fluidigmc1 Embedding Analysis")
print("="*60)

fluidigmc1_emb = label_embeddings['fluidigmc1']
null_emb = label_embeddings['null']

# 检查embedding的各个维度
print(f"\\nfluidigmc1 embedding stats:")
print(f"  Mean:    {fluidigmc1_emb.mean():.4f}")
print(f"  Std:     {fluidigmc1_emb.std():.4f}")
print(f"  Min:     {fluidigmc1_emb.min():.4f}")
print(f"  Max:     {fluidigmc1_emb.max():.4f}")

# 对比其他batch
other_means = [label_embeddings[b].mean() for b in le.classes_ if b != 'fluidigmc1']
other_stds = [label_embeddings[b].std() for b in le.classes_ if b != 'fluidigmc1']
print(f"\\nOther batches average:")
print(f"  Mean of means: {np.mean(other_means):.4f}")
print(f"  Mean of stds:  {np.mean(other_stds):.4f}")

if abs(fluidigmc1_emb.mean() - np.mean(other_means)) > 2 * np.std(other_means):
    print("\\n⚠️  fluidigmc1 embedding mean is significantly different!")
if abs(fluidigmc1_emb.std() - np.mean(other_stds)) > 2 * np.std(other_stds):
    print("⚠️  fluidigmc1 embedding std is significantly different!")
"""

# ============================================================
# Cell 6: Compare Sampling Trajectories
# ============================================================
diagnostic_cell_6 = """
# Cell 6: 比较采样轨迹

import torch
import numpy as np

# 获取diffusion模型
diffusion_model = diffusion.diffusion
model = diffusion.ema_model if diffusion.hparams.use_ema else diffusion.model
diffusion_model.model = model
device = next(model.parameters()).device

le = LabelEncoder()
le.fit(batch_labels)

print("="*60)
print("Sampling Trajectory Comparison")
print("="*60)

# 固定起始噪声
torch.manual_seed(42)
n_test = 10
x_start = torch.randn(n_test, diffusion.hparams.input_dim, device=device)

# 采样并记录轨迹
def sample_with_trajectory(labels, guidance_scale):
    x = x_start.clone()
    trajectory = [x.cpu().numpy().copy()]
    
    sampling_timesteps = 50
    total_timesteps = diffusion_model.num_timesteps
    
    # 创建采样时间步
    times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))
    
    for time, time_next in time_pairs:
        t = torch.full((n_test,), time, device=device, dtype=torch.long)
        
        # 模型预测
        model_output, null_output = model.forward_with_cond_scale(
            x, t, labels=labels, cond_scale=guidance_scale, rescaled_phi=0.7
        )
        
        # 预测x_start
        x_start_pred = diffusion_model.predict_start_from_noise(x, t, model_output)
        
        # DDIM step (简化版)
        if time_next < 0:
            x = x_start_pred
        else:
            alpha = diffusion_model.alphas_cumprod[time]
            alpha_next = diffusion_model.alphas_cumprod[time_next]
            sigma = 0.0  # DDIM eta=0
            c = (1 - alpha_next - sigma**2).sqrt()
            x = x_start_pred * alpha_next.sqrt() + c * model_output
        
        trajectory.append(x.cpu().numpy().copy())
    
    return trajectory

# 测试不同情况
with torch.no_grad():
    # 无条件采样
    traj_uncond = sample_with_trajectory(None, 0.0)
    
    # fluidigmc1条件采样 (guidance_scale=0.0)
    fluidigmc1_idx = le.transform(['fluidigmc1'])[0]
    fluidigmc1_labels = torch.full((n_test,), fluidigmc1_idx, device=device, dtype=torch.long)
    traj_fluid_0 = sample_with_trajectory(fluidigmc1_labels, 0.0)
    
    # celseq2条件采样 (guidance_scale=0.0)
    celseq2_idx = le.transform(['celseq2'])[0] if 'celseq2' in le.classes_ else 0
    celseq2_labels = torch.full((n_test,), celseq2_idx, device=device, dtype=torch.long)
    traj_celseq2_0 = sample_with_trajectory(celseq2_labels, 0.0)

# 分析轨迹差异
print(f"\\nTrajectory comparison (mean over {n_test} samples):")
print(f"\\n{'Step':>5} {'Uncond Norm':>12} {'Fluid Norm':>12} {'Celseq2 Norm':>14} {'Fluid-Uncond':>14} {'Celseq2-Uncond':>16}")
print("-" * 85)

steps_to_show = [0, 10, 20, 30, 40, 50]
for step in steps_to_show:
    uncond_norm = np.linalg.norm(traj_uncond[step], axis=1).mean()
    fluid_norm = np.linalg.norm(traj_fluid_0[step], axis=1).mean()
    celseq2_norm = np.linalg.norm(traj_celseq2_0[step], axis=1).mean()
    
    fluid_diff = np.linalg.norm(traj_fluid_0[step] - traj_uncond[step], axis=1).mean()
    celseq2_diff = np.linalg.norm(traj_celseq2_0[step] - traj_uncond[step], axis=1).mean()
    
    print(f"{step:>5} {uncond_norm:>12.4f} {fluid_norm:>12.4f} {celseq2_norm:>14.4f} {fluid_diff:>14.4f} {celseq2_diff:>16.4f}")

# 可视化最终样本的差异
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 最终样本的L2范数分布
final_uncond = np.linalg.norm(traj_uncond[-1], axis=1)
final_fluid = np.linalg.norm(traj_fluid_0[-1], axis=1)
final_celseq2 = np.linalg.norm(traj_celseq2_0[-1], axis=1)

axes[0].hist(final_uncond, bins=20, alpha=0.5, label='Unconditional')
axes[0].hist(final_fluid, bins=20, alpha=0.5, label='fluidigmc1 (cfg=0)')
axes[0].hist(final_celseq2, bins=20, alpha=0.5, label='celseq2 (cfg=0)')
axes[0].set_xlabel('L2 Norm')
axes[0].set_ylabel('Count')
axes[0].set_title('Final Sample Norms')
axes[0].legend()

# 轨迹差异随时间变化
steps = range(len(traj_uncond))
fluid_diffs = [np.linalg.norm(traj_fluid_0[s] - traj_uncond[s], axis=1).mean() for s in steps]
celseq2_diffs = [np.linalg.norm(traj_celseq2_0[s] - traj_uncond[s], axis=1).mean() for s in steps]

axes[1].plot(steps, fluid_diffs, 'r-', label='fluidigmc1')
axes[1].plot(steps, celseq2_diffs, 'b-', label='celseq2')
axes[1].set_xlabel('DDIM Step')
axes[1].set_ylabel('|cond - uncond|')
axes[1].set_title('Trajectory Deviation from Unconditional')
axes[1].legend()

# 最终样本的第一维度分布
axes[2].hist(traj_uncond[-1][:, 0], bins=20, alpha=0.5, label='Unconditional')
axes[2].hist(traj_fluid_0[-1][:, 0], bins=20, alpha=0.5, label='fluidigmc1')
axes[2].hist(traj_celseq2_0[-1][:, 0], bins=20, alpha=0.5, label='celseq2')
axes[2].set_xlabel('Dim 0 Value')
axes[2].set_ylabel('Count')
axes[2].set_title('Final Sample Dim 0 Distribution')
axes[2].legend()

plt.tight_layout()
plt.show()

print("\\n" + "="*60)
print("Conclusion")
print("="*60)
if fluid_diffs[-1] > celseq2_diffs[-1] * 2:
    print(f"⚠️  fluidigmc1 deviates {fluid_diffs[-1]/celseq2_diffs[-1]:.1f}x more from unconditional than celseq2!")
    print("   This suggests the model has learned different behavior for fluidigmc1.")
"""

# Print instructions
print("""
CFG Diagnostic Tests Created!

复制以下cell到您的notebook运行：

1. Cell 1: 检查batch分布和基本统计
2. Cell 2: 测试模型对不同batch的forward输出  
3. Cell 3: 逐步测试CFG过程，找出问题所在
4. Cell 4: 深入分析fluidigmc1
5. Cell 5: 检查Label Embedding
6. Cell 6: 比较采样轨迹

建议按顺序运行，每个cell的输出会帮助定位问题。
""")

