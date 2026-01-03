# CFG诊断测试

请按顺序复制每个cell到notebook运行。

---

## Cell 1: 基本统计

```python
# Cell 1: 检查batch分布
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

print("="*60)
print("Batch Distribution")
print("="*60)

batch_counts = batch_labels.value_counts()
print(batch_counts)

# 特别关注fluidigmc1
print(f"\nfluidigmc1 样本数: {(batch_labels == 'fluidigmc1').sum()}")
print(f"总样本数: {len(batch_labels)}")
print(f"占比: {100 * (batch_labels == 'fluidigmc1').sum() / len(batch_labels):.2f}%")
```

---

## Cell 2: 核心诊断 - 逐步测试CFG

```python
# Cell 2: 核心诊断 - 找出问题所在
import torch
import numpy as np
from functools import partial
from sklearn.preprocessing import LabelEncoder

# 获取模型
model = diffusion.ema_model if diffusion.hparams.use_ema else diffusion.model
model.eval()
device = next(model.parameters()).device

# 创建测试输入
torch.manual_seed(42)
n_test = 100
x_test = torch.randn(n_test, diffusion.hparams.input_dim, device=device)
t_test = torch.full((n_test,), 500, device=device, dtype=torch.long)

le = LabelEncoder()
le.fit(batch_labels)

def std_fn(x):
    return torch.std(x, dim=tuple(range(1, x.ndim)), keepdim=True)

print("="*60)
print("CFG Step-by-Step Analysis (cond_scale=0.0)")
print("="*60)

rescaled_phi = 0.7
cond_scale = 0.0

all_results = {}

with torch.no_grad():
    # 无条件预测
    null_logits = model.forward(x_test, t_test, labels=None, cond_drop_prob=1.0)
    null_std = std_fn(null_logits)
    
    print(f"\nUnconditional:")
    print(f"  Std per sample (mean): {null_std.mean().item():.4f}")
    
    for batch_name in le.classes_:
        batch_idx = le.transform([batch_name])[0]
        labels = torch.full((n_test,), batch_idx, device=device, dtype=torch.long)
        
        # 条件预测
        logits = model.forward(x_test, t_test, labels=labels)
        logits_std = std_fn(logits)
        
        # CFG计算
        scaled_logits = null_logits + (logits - null_logits) * cond_scale  # = null_logits
        scaled_std = std_fn(scaled_logits)
        
        # Rescaling
        std_ratio = logits_std / (scaled_std + 1e-8)
        rescaled_logits = scaled_logits * std_ratio
        
        # Interpolation
        final = rescaled_logits * rescaled_phi + scaled_logits * (1.0 - rescaled_phi)
        
        # 差异
        diff_from_null = torch.norm(final - null_logits, dim=1).mean().item()
        
        all_results[batch_name] = {
            'logits_std': logits_std.mean().item(),
            'scaled_std': scaled_std.mean().item(),
            'std_ratio': std_ratio.mean().item(),
            'diff_from_null': diff_from_null,
        }

# 打印结果
print(f"\n{'Batch':<15} {'Cond Std':>10} {'Scaled Std':>12} {'Std Ratio':>10} {'|Final-Null|':>12}")
print("-" * 65)
for batch_name in le.classes_:
    r = all_results[batch_name]
    flag = "⚠️" if abs(r['std_ratio'] - 1.0) > 0.1 or r['diff_from_null'] > 0.01 else ""
    print(f"{batch_name:<15} {r['logits_std']:>10.4f} {r['scaled_std']:>12.4f} {r['std_ratio']:>10.4f} {r['diff_from_null']:>12.6f} {flag}")

# 找问题
print("\n" + "="*60)
print("Problem Detection")
print("="*60)

for batch_name in le.classes_:
    r = all_results[batch_name]
    issues = []
    if abs(r['std_ratio'] - 1.0) > 0.1:
        issues.append(f"std_ratio={r['std_ratio']:.3f} (should be ~1.0)")
    if r['diff_from_null'] > 0.01:
        issues.append(f"final differs from unconditional by {r['diff_from_null']:.6f}")
    if issues:
        print(f"⚠️  {batch_name}: {', '.join(issues)}")
```

---

## Cell 3: 可视化std ratio

```python
# Cell 3: 可视化std ratio问题
import matplotlib.pyplot as plt

batch_names = list(all_results.keys())
std_ratios = [all_results[b]['std_ratio'] for b in batch_names]
diff_from_nulls = [all_results[b]['diff_from_null'] for b in batch_names]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Std ratio
colors = ['red' if abs(r - 1.0) > 0.1 else 'blue' for r in std_ratios]
axes[0].bar(range(len(batch_names)), std_ratios, color=colors)
axes[0].axhline(1.0, color='green', linestyle='--', linewidth=2, label='Expected (1.0)')
axes[0].set_xticks(range(len(batch_names)))
axes[0].set_xticklabels(batch_names, rotation=45, ha='right')
axes[0].set_ylabel('std(logits) / std(scaled_logits)')
axes[0].set_title('Std Ratio by Batch\n(Red = Problem, Should be ~1.0 for cond_scale=0)')
axes[0].legend()

# Diff from null
colors = ['red' if d > 0.01 else 'blue' for d in diff_from_nulls]
axes[1].bar(range(len(batch_names)), diff_from_nulls, color=colors)
axes[1].axhline(0.0, color='green', linestyle='--', linewidth=2, label='Expected (0.0)')
axes[1].set_xticks(range(len(batch_names)))
axes[1].set_xticklabels(batch_names, rotation=45, ha='right')
axes[1].set_ylabel('|final - null_logits|')
axes[1].set_title('Deviation from Unconditional\n(Red = Problem, Should be ~0 for cond_scale=0)')
axes[1].legend()

plt.tight_layout()
plt.show()
```

---

## Cell 4: 检查Label Embedding

```python
# Cell 4: 检查Label Embedding
import torch
import numpy as np

model = diffusion.ema_model if diffusion.hparams.use_ema else diffusion.model
device = next(model.parameters()).device

le = LabelEncoder()
le.fit(batch_labels)

print("="*60)
print("Label Embedding Analysis")
print("="*60)

label_embeddings = {}
with torch.no_grad():
    # Null embedding
    null_emb = model.null_label_emb.detach().cpu().numpy()
    label_embeddings['null'] = null_emb
    
    for batch_name in le.classes_:
        batch_idx = le.transform([batch_name])[0]
        labels = torch.tensor([batch_idx], device=device, dtype=torch.long)
        emb = model.label_embedding(labels).detach().cpu().numpy()[0]
        label_embeddings[batch_name] = emb

# 计算与null的距离
print(f"\n{'Batch':<15} {'Emb Norm':>10} {'Dist from Null':>15} {'Cosine Sim':>12}")
print("-" * 55)

for batch_name in le.classes_:
    emb = label_embeddings[batch_name]
    dist = np.linalg.norm(emb - null_emb)
    cos_sim = np.dot(emb, null_emb) / (np.linalg.norm(emb) * np.linalg.norm(null_emb) + 1e-8)
    print(f"{batch_name:<15} {np.linalg.norm(emb):>10.4f} {dist:>15.4f} {cos_sim:>12.4f}")

# 检查异常
print("\n" + "="*60)
print("Embedding Anomaly Check")
print("="*60)

emb_norms = {b: np.linalg.norm(label_embeddings[b]) for b in le.classes_}
mean_norm = np.mean(list(emb_norms.values()))
std_norm = np.std(list(emb_norms.values()))

for batch_name in le.classes_:
    if abs(emb_norms[batch_name] - mean_norm) > 2 * std_norm:
        print(f"⚠️  {batch_name} embedding norm ({emb_norms[batch_name]:.4f}) is {abs(emb_norms[batch_name] - mean_norm)/std_norm:.1f} std away from mean!")
```

---

## Cell 5: 最关键的测试 - 直接对比采样

```python
# Cell 5: 直接对比采样结果
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(batch_labels)

# 固定随机种子
torch.manual_seed(123)

n_test = 500  # 测试500个样本

print("="*60)
print("Direct Sampling Comparison")
print("="*60)

diffusion.eval()
with torch.no_grad():
    # 测试1: 完全无条件 (labels=None)
    samples_none = diffusion.sample(
        num_samples=n_test,
        sampling_timesteps=50,
        labels=None,  # 关键：不提供labels
        ddim_sampling_eta=0,
        use_ema=True,
        clip_denoised=False
    ).cpu().numpy()
    
    # 测试2: fluidigmc1 + guidance_scale=0
    torch.manual_seed(123)  # 同样的随机种子！
    fluidigmc1_idx = le.transform(['fluidigmc1'])[0]
    fluidigmc1_labels = torch.full((n_test,), fluidigmc1_idx, dtype=torch.long, device=device)
    
    samples_fluid_0 = diffusion.sample(
        num_samples=n_test,
        sampling_timesteps=50,
        labels=fluidigmc1_labels,
        guidance_scale=0.0,
        ddim_sampling_eta=0,
        use_ema=True,
        clip_denoised=False
    ).cpu().numpy()
    
    # 测试3: celseq2 + guidance_scale=0 (作为参照)
    torch.manual_seed(123)
    celseq2_idx = le.transform(['celseq2'])[0] if 'celseq2' in le.classes_ else le.transform([le.classes_[1]])[0]
    celseq2_labels = torch.full((n_test,), celseq2_idx, dtype=torch.long, device=device)
    
    samples_celseq2_0 = diffusion.sample(
        num_samples=n_test,
        sampling_timesteps=50,
        labels=celseq2_labels,
        guidance_scale=0.0,
        ddim_sampling_eta=0,
        use_ema=True,
        clip_denoised=False
    ).cpu().numpy()

# 计算差异
diff_fluid = np.linalg.norm(samples_fluid_0 - samples_none, axis=1)
diff_celseq2 = np.linalg.norm(samples_celseq2_0 - samples_none, axis=1)

print(f"\n使用同样的随机种子，比较最终采样结果：")
print(f"\n|fluidigmc1(cfg=0) - unconditional|:")
print(f"  Mean: {diff_fluid.mean():.6f}")
print(f"  Max:  {diff_fluid.max():.6f}")

print(f"\n|celseq2(cfg=0) - unconditional|:")
print(f"  Mean: {diff_celseq2.mean():.6f}")
print(f"  Max:  {diff_celseq2.max():.6f}")

print(f"\n差异比例: fluidigmc1 / celseq2 = {diff_fluid.mean() / (diff_celseq2.mean() + 1e-8):.2f}x")

if diff_fluid.mean() > 0.01 or diff_celseq2.mean() > 0.01:
    print(f"\n⚠️  即使guidance_scale=0.0，采样结果仍与无条件采样不同！")
    print("   这证明了CFG实现中的rescaling步骤引入了条件信息。")
else:
    print(f"\n✓ 采样结果与无条件采样相同，问题可能在其他地方。")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(diff_fluid, bins=30, alpha=0.7, label='fluidigmc1')
axes[0].hist(diff_celseq2, bins=30, alpha=0.7, label='celseq2')
axes[0].set_xlabel('|sample(cfg=0) - sample(unconditional)|')
axes[0].set_ylabel('Count')
axes[0].set_title('Difference from Unconditional Sampling')
axes[0].legend()

# 各batch的最终样本范数
norms_none = np.linalg.norm(samples_none, axis=1)
norms_fluid = np.linalg.norm(samples_fluid_0, axis=1)
norms_celseq2 = np.linalg.norm(samples_celseq2_0, axis=1)

axes[1].boxplot([norms_none, norms_fluid, norms_celseq2], 
                labels=['Unconditional', 'fluidigmc1 (cfg=0)', 'celseq2 (cfg=0)'])
axes[1].set_ylabel('Sample L2 Norm')
axes[1].set_title('Final Sample Norms')

plt.tight_layout()
plt.show()
```

---

## Cell 6: 根本原因确认

```python
# Cell 6: 确认根本原因
print("="*60)
print("ROOT CAUSE ANALYSIS")
print("="*60)

print("""
如果 Cell 5 显示 fluidigmc1 和 celseq2 的差异都接近0：
→ 问题不在CFG实现，可能在其他地方

如果 Cell 5 显示有差异（即使 guidance_scale=0）：
→ 问题在 forward_with_cond_scale 的 rescaling 步骤

检查代码 diffusion_model.py 第 426-434 行：
```python
rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))
final = rescaled_logits * rescaled_phi + scaled_logits * (1.0 - rescaled_phi)
```

当 cond_scale=0 时：
- scaled_logits = null_logits（正确）
- 但 rescaled_logits 使用了 std(logits)（条件预测的std）来rescale
- 这导致条件信息通过std泄露到最终结果！

对于fluidigmc1，如果其条件预测的std与无条件预测的std差异大：
→ std_ratio = std(logits)/std(null_logits) ≠ 1
→ rescaling会改变结果
→ 最终输出不等于null_logits
""")
```

