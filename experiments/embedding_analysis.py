"""
为什么fluidigmc1的embedding异常？深入分析

复制到notebook运行
"""

embedding_analysis = '''
# ============================================================
# 深入分析：为什么fluidigmc1的embedding异常？
# ============================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

model = diffusion.ema_model if diffusion.hparams.use_ema else diffusion.model
model.eval()
device = next(model.parameters()).device

le = LabelEncoder()
le.fit(batch_labels)

print("="*70)
print("为什么fluidigmc1的embedding异常？深入分析")
print("="*70)

# ============================================================
# 1. 检查样本数量与embedding范数的关系
# ============================================================
print("\\n" + "="*50)
print("1. 样本数量 vs Embedding范数")
print("="*50)

batch_counts = batch_labels.value_counts()
label_embeddings = {}
embedding_norms = {}

with torch.no_grad():
    null_emb = model.null_label_emb.detach().cpu().numpy()
    
    for batch_name in le.classes_:
        batch_idx = le.transform([batch_name])[0]
        labels = torch.tensor([batch_idx], device=device, dtype=torch.long)
        emb = model.label_embedding(labels).detach().cpu().numpy()[0]
        label_embeddings[batch_name] = emb
        embedding_norms[batch_name] = np.linalg.norm(emb)

# 创建对比表
print(f"\\n{'Batch':<15} {'样本数':>8} {'占比%':>8} {'Emb Norm':>10} {'与Null距离':>12}")
print("-" * 60)
for batch_name in sorted(le.classes_, key=lambda x: batch_counts.get(x, 0)):
    count = batch_counts.get(batch_name, 0)
    pct = 100 * count / len(batch_labels)
    norm = embedding_norms[batch_name]
    dist = np.linalg.norm(label_embeddings[batch_name] - null_emb)
    flag = "⚠️" if batch_name == 'fluidigmc1' else ""
    print(f"{batch_name:<15} {count:>8} {pct:>8.2f} {norm:>10.4f} {dist:>12.4f} {flag}")

# 可视化：样本数 vs embedding范数
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

counts = [batch_counts.get(b, 0) for b in le.classes_]
norms = [embedding_norms[b] for b in le.classes_]

colors = ['red' if b == 'fluidigmc1' else 'blue' for b in le.classes_]
axes[0].scatter(counts, norms, c=colors, s=100)
for i, b in enumerate(le.classes_):
    axes[0].annotate(b, (counts[i], norms[i]), fontsize=8)
axes[0].set_xlabel('Sample Count')
axes[0].set_ylabel('Embedding Norm')
axes[0].set_title('Sample Count vs Embedding Norm\\n(Red = fluidigmc1)')

# 相关性
corr = np.corrcoef(counts, norms)[0, 1]
print(f"\\n样本数与embedding范数的相关系数: {corr:.4f}")
if corr < -0.3:
    print("⚠️  负相关：样本越少，embedding范数越大！这说明小样本类别的embedding学习不充分。")

# ============================================================
# 2. 检查Label Embedding的输入权重
# ============================================================
print("\\n" + "="*50)
print("2. Label Embedding网络权重分析")
print("="*50)

# 获取第一层Linear的权重
first_linear = model.label_embedding.net[0]
W = first_linear.weight.detach().cpu().numpy()  # [hidden_dim, num_classes]
b = first_linear.bias.detach().cpu().numpy()    # [hidden_dim]

print(f"\\n第一层Linear权重 shape: {W.shape}")
print(f"num_classes = {W.shape[1]}, hidden_dim = {W.shape[0]}")

# 每个类别对应的权重列的范数
weight_norms_per_class = np.linalg.norm(W, axis=0)  # [num_classes]
print(f"\\n每个batch对应的权重列范数:")
print(f"{'Batch':<15} {'Label Index':>12} {'Weight Norm':>12}")
print("-" * 42)
for batch_name in le.classes_:
    batch_idx = le.transform([batch_name])[0]
    w_norm = weight_norms_per_class[batch_idx]
    flag = "⚠️" if batch_name == 'fluidigmc1' else ""
    print(f"{batch_name:<15} {batch_idx:>12} {w_norm:>12.4f} {flag}")

# 可视化权重
axes[1].bar(range(len(le.classes_)), weight_norms_per_class[:len(le.classes_)],
           color=['red' if i == le.transform(['fluidigmc1'])[0] else 'blue' for i in range(len(le.classes_))])
axes[1].set_xticks(range(len(le.classes_)))
axes[1].set_xticklabels(le.classes_, rotation=45, ha='right')
axes[1].set_ylabel('Weight Column Norm')
axes[1].set_title('First Linear Layer Weight Norms per Batch')

plt.tight_layout()
plt.show()

# ============================================================
# 3. 检查训练样本分布的影响
# ============================================================
print("\\n" + "="*50)
print("3. 假设验证：样本不平衡导致embedding异常")
print("="*50)

# 计算：如果embedding范数与样本数成反比
expected_norm_if_inverse = 1.0 / np.array(counts)
expected_norm_if_inverse = expected_norm_if_inverse / expected_norm_if_inverse.max() * max(norms)

print("\\n如果embedding范数与样本数成反比（越少样本，范数越大）：")
print("这可能是因为小样本类别的梯度更新不够稳定，embedding没有很好地收敛。")

# ============================================================
# 4. 检查one-hot编码后的输入
# ============================================================
print("\\n" + "="*50)
print("4. One-Hot编码分析")
print("="*50)

print("\\nLabelEmbedding使用one-hot编码：")
print("  label -> [0,0,...,1,...,0] (只有对应位置为1)")
print("  然后经过 Linear -> SiLU -> Linear")

print("\\n对于fluidigmc1 (假设index=2):")
fluidigmc1_idx = le.transform(['fluidigmc1'])[0]
print(f"  实际index: {fluidigmc1_idx}")
print(f"  one-hot: {F.one_hot(torch.tensor([fluidigmc1_idx]), num_classes=len(le.classes_)).numpy()}")

# 第一层输出 = W[:, fluidigmc1_idx] + b
first_layer_out = W[:, fluidigmc1_idx] + b
print(f"\\n第一层输出（Linear后，SiLU前）:")
print(f"  Mean: {first_layer_out.mean():.4f}")
print(f"  Std:  {first_layer_out.std():.4f}")
print(f"  Norm: {np.linalg.norm(first_layer_out):.4f}")

# 对比其他batch
print(f"\\n各batch第一层输出范数:")
first_layer_norms = {}
for batch_name in le.classes_:
    idx = le.transform([batch_name])[0]
    out = W[:, idx] + b
    first_layer_norms[batch_name] = np.linalg.norm(out)
    
for batch_name in sorted(first_layer_norms.keys(), key=lambda x: first_layer_norms[x], reverse=True):
    flag = "⚠️" if batch_name == 'fluidigmc1' else ""
    print(f"  {batch_name:<15}: {first_layer_norms[batch_name]:.4f} {flag}")

# ============================================================
# 5. 结论
# ============================================================
print("\\n" + "="*50)
print("5. 结论")
print("="*50)

print("""
fluidigmc1 embedding异常的可能原因：

1. 【样本不平衡】
   - fluidigmc1只有638个样本（3.89%），是最少的batch
   - 训练过程中，这个batch的样本被采样的次数相对较少
   - 梯度更新不够充分，导致embedding没有很好地收敛

2. 【梯度方向不稳定】
   - 小样本类别的梯度方向更容易受到个别样本的影响
   - 可能导致权重向量指向一个异常的方向

3. 【没有权重正则化】
   - embedding网络没有对权重范数进行约束
   - 小样本类别的权重可能在训练中"漂移"

4. 【随机初始化】
   - 如果初始化时fluidigmc1对应的权重碰巧范数较大
   - 由于更新次数少，可能没有机会调整到正常范围

解决方案：
1. 数据增强或过采样，增加fluidigmc1的样本数
2. 使用balanced batch sampler，确保每个batch的类别均匀
3. 添加embedding正则化（如weight decay只作用于embedding层）
4. 重新训练模型，观察embedding的变化
""")
'''

print(embedding_analysis)

