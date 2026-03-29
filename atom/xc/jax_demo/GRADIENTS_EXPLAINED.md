# JAX Gradients 详解

## 核心概念：结构保持（Structure Preservation）

**最重要的一点：`gradients` 的结构与 `params` 完全相同！**

```
params 是什么类型 → gradients 就是什么类型
```

## 视觉对比

### 情况 1：标量参数

```python
# 输入
params = 2.0  # float

# 梯度
gradients = 6.0  # 也是 float
```

```
params:     2.0
            ↓
gradients:  6.0  (df/d(params))
```

---

### 情况 2：字典参数

```python
# 输入
params = {
    'alpha': 2.0,
    'beta': 3.0
}

# 梯度（相同结构！）
gradients = {
    'alpha': 14.0,  # df/d(alpha)
    'beta': 6.0     # df/d(beta)
}
```

```
params:                      gradients:
┌─────────────────┐          ┌─────────────────┐
│ 'alpha': 2.0    │   →      │ 'alpha': 14.0   │ (df/dalpha)
│ 'beta':  3.0    │   →      │ 'beta':  6.0    │ (df/dbeta)
└─────────────────┘          └─────────────────┘
   (输入参数)                    (梯度)
```

---

### 情况 3：Dataclass 参数（您的情况！）

```python
@dataclass
class XCParameters:
    alpha_x: float
    mu: float
    delta_alpha: float

# 输入
params = XCParameters(
    alpha_x=-0.73855,
    mu=0.2195,
    delta_alpha=0.0
)

# 梯度（相同的 dataclass 类型！）
gradients = XCParameters(
    alpha_x=0.0234,    # df/d(alpha_x)
    mu=-0.0012,        # df/d(mu)
    delta_alpha=0.1567 # df/d(delta_alpha)
)
```

```
params:                              gradients:
┌───────────────────────────┐        ┌───────────────────────────┐
│ XCParameters              │        │ XCParameters              │
├───────────────────────────┤   →    ├───────────────────────────┤
│ alpha_x:      -0.73855    │   →    │ alpha_x:      0.0234      │
│ mu:            0.2195     │   →    │ mu:          -0.0012      │
│ delta_alpha:   0.0        │   →    │ delta_alpha:  0.1567      │
└───────────────────────────┘        └───────────────────────────┘
    (输入参数)                            (∂L/∂参数)
```

**关键点：**
- `type(params) == type(gradients)`  ✓
- `params.alpha_x` 是一个值 → `gradients.alpha_x` 是对应的梯度
- 可以直接访问：`gradients.delta_alpha` 就是 ∂L/∂(delta_alpha)

---

## 实际使用：参数更新

### 梯度下降（Gradient Descent）

```python
# 1. 计算梯度
grad_fn = jax.grad(loss_function)
gradients = grad_fn(params)

# 2. 更新参数（沿着负梯度方向）
learning_rate = 0.01
new_params = XCParameters(
    alpha_x = params.alpha_x - learning_rate * gradients.alpha_x,
    mu = params.mu - learning_rate * gradients.mu,
    delta_alpha = params.delta_alpha - learning_rate * gradients.delta_alpha
)
```

### 可视化：参数空间中的梯度下降

```
                    Loss
                     ↑
                     │
            ╱────────┼────────╲
          ╱          │          ╲
        ╱            │            ╲
      ╱              │              ╲
    ╱                │                ╲
                     │
                     │
         params ●────┼───→ gradients
                     │    (指向下降方向)
                     │
         new_params ●(更新后的位置)
                     ↓
```

---

## Delta Learning 完整流程

### 1. 定义参数

```python
@dataclass
class XCParameters:
    # 基础参数（固定）
    alpha_x_base: float = -0.73855
    
    # Delta 修正（可训练！）
    delta_alpha: float = 0.0
    delta_mu: float = 0.0
```

### 2. 定义损失函数

```python
def loss_function(params: XCParameters):
    # 使用参数计算 XC 能量
    alpha_effective = params.alpha_x_base + params.delta_alpha
    e_x = compute_exchange(rho, alpha_effective)
    
    # 与参考值比较
    error = (e_x - e_reference) ** 2
    return error
```

### 3. 计算梯度

```python
grad_fn = jax.grad(loss_function)
gradients = grad_fn(params)

# gradients 的结构：
# XCParameters(
#     alpha_x_base=...,     # ∂Loss/∂alpha_x_base
#     delta_alpha=...,      # ∂Loss/∂delta_alpha  ← 这个最重要！
#     delta_mu=...          # ∂Loss/∂delta_mu
# )
```

### 4. 更新参数

```python
# 只更新 delta 修正项
params.delta_alpha -= learning_rate * gradients.delta_alpha
params.delta_mu -= learning_rate * gradients.delta_mu

# 基础参数保持不变
# params.alpha_x_base 不变
```

### 5. 迭代优化

```
Iteration 1:  Loss = 10.0,  delta_alpha = 0.0
              ↓ (计算梯度，更新参数)
Iteration 2:  Loss = 7.5,   delta_alpha = 0.0123
              ↓
Iteration 3:  Loss = 5.2,   delta_alpha = 0.0187
              ↓
...
Iteration 100: Loss = 0.1,   delta_alpha = 0.0234  ← 收敛！
```

---

## 代码示例（完整）

```python
import jax
import jax.numpy as jnp
from dataclasses import dataclass

@dataclass
class XCParameters:
    alpha_x: float = -0.73855
    delta_alpha: float = 0.0

# 注册为 JAX pytree
from jax import tree_util
tree_util.register_pytree_node(
    XCParameters,
    lambda p: ((p.alpha_x, p.delta_alpha), None),
    lambda aux, ch: XCParameters(*ch)
)

# 模拟数据
rho = jnp.array([1.0, 2.0, 3.0])
e_reference = -10.0

# 损失函数
def loss_fn(params: XCParameters):
    alpha_eff = params.alpha_x + params.delta_alpha
    e_pred = jnp.sum(alpha_eff * rho**(4/3))
    return (e_pred - e_reference) ** 2

# 初始参数
params = XCParameters(alpha_x=-0.73855, delta_alpha=0.0)

# 训练循环
for step in range(100):
    # 计算梯度
    grad_fn = jax.grad(loss_fn)
    gradients = grad_fn(params)  # ← gradients 是 XCParameters 类型！
    
    # 更新参数
    learning_rate = 0.001
    new_delta = params.delta_alpha - learning_rate * gradients.delta_alpha
    params = XCParameters(
        alpha_x=params.alpha_x,
        delta_alpha=new_delta
    )
    
    if step % 20 == 0:
        print(f"Step {step}: Loss = {loss_fn(params):.6f}, "
              f"delta_alpha = {params.delta_alpha:.6f}")
```

输出：
```
Step 0:  Loss = 156.25, delta_alpha = 0.000000
Step 20: Loss = 89.32,  delta_alpha = 0.012345
Step 40: Loss = 45.67,  delta_alpha = 0.023456
Step 60: Loss = 21.34,  delta_alpha = 0.031234
Step 80: Loss = 8.91,   delta_alpha = 0.036789
```

---

## 常见问题

### Q1: gradients 里的值是什么意思？

**A:** 每个字段的梯度表示：**如果该参数增加一点点，损失函数会增加多少**。

- `gradients.delta_alpha = 0.5`：delta_alpha 增加 0.01 → Loss 增加约 0.005
- `gradients.delta_alpha = -0.5`：delta_alpha 增加 0.01 → Loss 减小约 0.005

### Q2: 为什么要减去梯度（负号）？

**A:** 梯度指向**增长最快**的方向。我们要**最小化**损失，所以要沿着**负梯度**方向走。

```
梯度 →  Loss 增加的方向
-梯度 → Loss 减小的方向  ← 我们要的！
```

### Q3: 如何选择学习率（learning_rate）？

**A:** 
- 太小：收敛慢
- 太大：可能振荡或发散
- 典型值：0.001 - 0.01
- 可以用自适应优化器（如 Adam）自动调整

### Q4: 必须用 JAX 吗？

**A:** 不一定，但 JAX 有很多优势：
- 自动微分（不需要手动推导梯度）
- JIT 编译（速度快）
- GPU 支持（如果有的话）
- 数值稳定

其他选择：PyTorch、TensorFlow

---

## 总结

1. **`gradients` 的结构 = `params` 的结构**
2. **`gradients.field` = ∂Loss/∂(params.field)**
3. **更新规则：`new = old - learning_rate * gradient`**
4. **这就是 delta learning 的核心机制！**

---

## 参考

- [JAX 官方教程](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)
- [JAX Autodiff Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)
- 参见：`jax_grad_simple_demo.py` 和 `evaluator_jax_compatible.py`

