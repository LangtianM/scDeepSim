"""
关键诊断测试 - 直接复制到notebook运行
"""

critical_test = '''
# 关键诊断：逐样本分析std_ratio
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 获取模型
model = diffusion.ema_model if diffusion.hparams.use_ema else diffusion.model
model.eval()
device = next(model.parameters()).device

le = LabelEncoder()
le.fit(batch_labels)

# 使用真实数据分布的起始点（不是随机噪声）
torch.manual_seed(42)
n_test = 500
x_test = torch.randn(n_test, diffusion.hparams.input_dim, device=device)
t_test = torch.full((n_test,), 500, device=device, dtype=torch.long)

def std_per_sample(x):
    """计算每个样本在feature维度上的std"""
    return torch.std(x, dim=1, keepdim=True)

print("="*60)
print("Per-Sample Std Ratio Analysis")
print("="*60)

with torch.no_grad():
    # 无条件预测
    null_logits = model.forward(x_test, t_test, labels=None, cond_drop_prob=1.0)
    null_std = std_per_sample(null_logits)  # [n_test, 1]
    
    # fluidigmc1条件预测
    fluidigmc1_idx = le.transform(['fluidigmc1'])[0]
    fluidigmc1_labels = torch.full((n_test,), fluidigmc1_idx, device=device, dtype=torch.long)
    fluid_logits = model.forward(x_test, t_test, labels=fluidigmc1_labels)
    fluid_std = std_per_sample(fluid_logits)
    
    # celseq2条件预测（参照）
    celseq2_idx = le.transform(['celseq2'])[0] if 'celseq2' in le.classes_ else le.transform([le.classes_[1]])[0]
    celseq2_labels = torch.full((n_test,), celseq2_idx, device=device, dtype=torch.long)
    celseq2_logits = model.forward(x_test, t_test, labels=celseq2_labels)
    celseq2_std = std_per_sample(celseq2_logits)
    
    # 计算std ratio
    fluid_std_ratio = (fluid_std / (null_std + 1e-8)).squeeze().cpu().numpy()
    celseq2_std_ratio = (celseq2_std / (null_std + 1e-8)).squeeze().cpu().numpy()

# 统计
print(f"\nfluidigmc1 std_ratio 统计:")
print(f"  Mean:   {fluid_std_ratio.mean():.4f}")
print(f"  Std:    {fluid_std_ratio.std():.4f}")
print(f"  Min:    {fluid_std_ratio.min():.4f}")
print(f"  Max:    {fluid_std_ratio.max():.4f}")
print(f"  >1.5:   {(fluid_std_ratio > 1.5).sum()} 个样本 ({100*(fluid_std_ratio > 1.5).mean():.1f}%)")
print(f"  <0.7:   {(fluid_std_ratio < 0.7).sum()} 个样本 ({100*(fluid_std_ratio < 0.7).mean():.1f}%)")

print(f"\ncelseq2 std_ratio 统计:")
print(f"  Mean:   {celseq2_std_ratio.mean():.4f}")
print(f"  Std:    {celseq2_std_ratio.std():.4f}")
print(f"  Min:    {celseq2_std_ratio.min():.4f}")
print(f"  Max:    {celseq2_std_ratio.max():.4f}")
print(f"  >1.5:   {(celseq2_std_ratio > 1.5).sum()} 个样本 ({100*(celseq2_std_ratio > 1.5).mean():.1f}%)")
print(f"  <0.7:   {(celseq2_std_ratio < 0.7).sum()} 个样本 ({100*(celseq2_std_ratio < 0.7).mean():.1f}%)")

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Histogram of std ratios
axes[0].hist(fluid_std_ratio, bins=50, alpha=0.7, label='fluidigmc1', color='red')
axes[0].hist(celseq2_std_ratio, bins=50, alpha=0.7, label='celseq2', color='blue')
axes[0].axvline(1.0, color='green', linestyle='--', linewidth=2, label='Expected (1.0)')
axes[0].set_xlabel('std(cond) / std(uncond)')
axes[0].set_ylabel('Count')
axes[0].set_title('Per-Sample Std Ratio Distribution')
axes[0].legend()

# Scatter plot
axes[1].scatter(range(n_test), fluid_std_ratio, alpha=0.5, s=10, label='fluidigmc1', c='red')
axes[1].scatter(range(n_test), celseq2_std_ratio, alpha=0.5, s=10, label='celseq2', c='blue')
axes[1].axhline(1.0, color='green', linestyle='--', linewidth=2)
axes[1].axhline(1.5, color='orange', linestyle=':', linewidth=1)
axes[1].axhline(0.7, color='orange', linestyle=':', linewidth=1)
axes[1].set_xlabel('Sample Index')
axes[1].set_ylabel('Std Ratio')
axes[1].set_title('Std Ratio per Sample')
axes[1].legend()

# Box plot
axes[2].boxplot([fluid_std_ratio, celseq2_std_ratio], labels=['fluidigmc1', 'celseq2'])
axes[2].axhline(1.0, color='green', linestyle='--', linewidth=2)
axes[2].set_ylabel('Std Ratio')
axes[2].set_title('Std Ratio Distribution')

plt.tight_layout()
plt.show()

# 分析rescaling的影响
print("\\n" + "="*60)
print("Rescaling Impact Analysis")
print("="*60)

# 计算最终输出（模拟CFG过程）
rescaled_phi = 0.7
cond_scale = 0.0

with torch.no_grad():
    # scaled_logits = null_logits (因为 cond_scale=0)
    scaled_logits = null_logits.clone()
    
    # fluidigmc1的rescaling
    fluid_rescaled = scaled_logits * (fluid_std / (null_std + 1e-8))
    fluid_final = fluid_rescaled * rescaled_phi + scaled_logits * (1.0 - rescaled_phi)
    fluid_diff = torch.norm(fluid_final - null_logits, dim=1).cpu().numpy()
    
    # celseq2的rescaling
    celseq2_rescaled = scaled_logits * (celseq2_std / (null_std + 1e-8))
    celseq2_final = celseq2_rescaled * rescaled_phi + scaled_logits * (1.0 - rescaled_phi)
    celseq2_diff = torch.norm(celseq2_final - null_logits, dim=1).cpu().numpy()

print(f"\\nfluidigmc1 |final - null|:")
print(f"  Mean: {fluid_diff.mean():.6f}")
print(f"  Max:  {fluid_diff.max():.6f}")

print(f"\\ncelseq2 |final - null|:")
print(f"  Mean: {celseq2_diff.mean():.6f}")
print(f"  Max:  {celseq2_diff.max():.6f}")

if fluid_diff.mean() > 0.01:
    print(f"\\n⚠️  fluidigmc1的最终输出与无条件输出有显著差异！")
    print(f"   这是因为std_ratio的variance导致rescaling不一致。")

# 检查异常样本
print("\\n" + "="*60)
print("Outlier Sample Analysis")
print("="*60)

# 找出std_ratio最极端的样本
extreme_idx = np.argmax(np.abs(fluid_std_ratio - 1.0))
print(f"\\n最极端的样本 (index {extreme_idx}):")
print(f"  fluidigmc1 std_ratio: {fluid_std_ratio[extreme_idx]:.4f}")
print(f"  celseq2 std_ratio:    {celseq2_std_ratio[extreme_idx]:.4f}")
print(f"  fluidigmc1 |final-null|: {fluid_diff[extreme_idx]:.6f}")
print(f"  celseq2 |final-null|:    {celseq2_diff[extreme_idx]:.6f}")
'''

print(critical_test)

