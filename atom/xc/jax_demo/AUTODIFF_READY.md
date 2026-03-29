# 参数自动微分就绪指南

## ✅ 设计完成

所有 XC 泛函参数现在都可以通过 autodiff 优化！

---

## 核心设计

### LDA 参数（lda.py）

```python
@dataclass
class LDAParameters(XCParameters):
    # Slater exchange
    C_x: float    # 可优化！
    
    # VWN correlation
    A: float      # 可优化！
    b: float      # 可优化！
    c: float      # 可优化！
    y0: float     # 可优化！

class LDA_SVWN:
    def _default_params(self):
        return LDAParameters(
            functional_name='LDA_SVWN',
            C_x=-(3/4) * (3/np.pi)**(1/3),  # 标准值
            A=0.0621814,                     # 标准值
            b=3.72744,                       # 标准值
            c=12.9352,                       # 标准值
            y0=-0.10498                      # 标准值
        )
    
    def compute_correlation_generic(self, density_data):
        # 从 params 获取（而不是硬编码！）
        A = self.params.A    # ← 可微分
        b = self.params.b    # ← 可微分
        c = self.params.c    # ← 可微分
        y0 = self.params.y0  # ← 可微分
        
        # 计算
        e_c = ... (使用 A, b, c, y0)
        v_c = ...
```

---

## JAX 自动微分示例

### 示例 1：优化单个 VWN 参数

```python
import jax
import jax.numpy as jnp
from delta.atomic_dft.xc.lda import LDA_SVWN, LDAParameters

# 注册 LDAParameters 为 JAX pytree
from jax import tree_util

tree_util.register_pytree_node(
    LDAParameters,
    # Flatten: 所有数值参数都是可微分的
    lambda p: ((p.C_x, p.A, p.b, p.c, p.y0), {'functional_name': p.functional_name}),
    # Unflatten
    lambda aux, ch: LDAParameters(
        functional_name=aux['functional_name'],
        C_x=ch[0], A=ch[1], b=ch[2], c=ch[3], y0=ch[4]
    )
)

# 定义损失函数
def loss_fn(params: LDAParameters):
    """计算预测能量与参考能量的误差"""
    evaluator = LDA_SVWN(params=params)
    density_data = DensityData(rho=rho)
    result = evaluator.compute_xc(density_data)
    
    # 计算总关联能
    E_c_pred = jnp.sum(rho * result.e_c * weights)
    E_c_ref = -5.0  # 参考值（高精度计算）
    
    return (E_c_pred - E_c_ref) ** 2

# 计算梯度
params = LDAParameters(
    functional_name='LDA_SVWN',
    C_x=-0.738559,
    A=0.0621814,   # 初始值
    b=3.72744,     # 初始值
    c=12.9352,     # 初始值
    y0=-0.10498    # 初始值
)

grad_fn = jax.grad(loss_fn)
gradients = grad_fn(params)

print(f"∂L/∂A  = {gradients.A}")
print(f"∂L/∂b  = {gradients.b}")
print(f"∂L/∂c  = {gradients.c}")
print(f"∂L/∂y0 = {gradients.y0}")

# 优化参数
learning_rate = 0.001
optimized_params = LDAParameters(
    functional_name='LDA_SVWN',
    C_x=params.C_x,
    A=params.A - learning_rate * gradients.A,      # 更新！
    b=params.b - learning_rate * gradients.b,      # 更新！
    c=params.c - learning_rate * gradients.c,      # 更新！
    y0=params.y0 - learning_rate * gradients.y0    # 更新！
)
```

---

### 示例 2：同时优化交换和关联

```python
def total_loss(params: LDAParameters):
    """优化交换和关联的所有参数"""
    evaluator = LDA_SVWN(params=params)
    result = evaluator.compute_xc(density_data)
    
    # 总 XC 能量
    E_xc_pred = jnp.sum(rho * (result.e_x + result.e_c) * weights)
    E_xc_ref = -12.5
    
    return (E_xc_pred - E_xc_ref) ** 2

# 计算梯度
grad_fn = jax.grad(total_loss)
gradients = grad_fn(params)

# 现在可以优化所有 5 个参数！
print(f"∂L/∂C_x = {gradients.C_x}")  # 交换参数
print(f"∂L/∂A   = {gradients.A}")    # 关联参数
print(f"∂L/∂b   = {gradients.b}")
print(f"∂L/∂c   = {gradients.c}")
print(f"∂L/∂y0  = {gradients.y0}")
```

---

### 示例 3：使用 Optax 优化器

```python
import optax

# 初始参数
params = LDAParameters(
    functional_name='LDA_SVWN',
    C_x=-0.738559,
    A=0.0621814,
    b=3.72744,
    c=12.9352,
    y0=-0.10498
)

# 设置优化器（Adam）
optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)

# 训练循环
for step in range(100):
    # 计算损失和梯度
    loss, grads = jax.value_and_grad(total_loss)(params)
    
    # 更新参数
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    if step % 10 == 0:
        print(f"Step {step}: Loss = {loss:.6f}")
        print(f"  A = {params.A:.6f}")

# 优化后的参数
print(f"\nOptimized parameters:")
print(f"  C_x = {params.C_x:.6f}")
print(f"  A   = {params.A:.6f}")
print(f"  b   = {params.b:.6f}")
print(f"  c   = {params.c:.6f}")
print(f"  y0  = {params.y0:.6f}")
```

---

## GGA 参数（gga_pbe.py）

### 参数定义

```python
@dataclass
class PBEParameters(XCParameters):
    mu: float     # 可优化！
    kappa: float  # 可优化！

class GGA_PBE:
    def _default_params(self):
        return PBEParameters(
            functional_name='GGA_PBE',
            mu=0.2195149727645171,  # 标准值
            kappa=0.804              # 标准值
        )
```

### 优化示例

```python
def pbe_loss(params: PBEParameters):
    evaluator = GGA_PBE(derivative_matrix=D, params=params)
    result = evaluator.compute_xc(density_data)
    # ...
    return loss

# 对 mu 和 kappa 求导
gradients = jax.grad(pbe_loss)(params)
print(f"∂L/∂mu    = {gradients.mu}")
print(f"∂L/∂kappa = {gradients.kappa}")
```

---

## 关键优势

### 1. 所有参数都可优化

```python
# LDA: 5 个可优化参数
LDAParameters:
  C_x   ← 交换
  A     ← 关联
  b     ← 关联
  c     ← 关联
  y0    ← 关联

# PBE: 2 个可优化参数  
PBEParameters:
  mu    ← 梯度增强
  kappa ← 梯度增强
```

### 2. 无需手动推导梯度

```python
# JAX 自动计算 ∂e_c/∂A, ∂e_c/∂b, ∂e_c/∂c, ∂e_c/∂y0
# 您不需要推导复杂的解析导数！

# 原始代码
e_c = A/2 * (ln(...) + 2*b/Q*arctan(...) - ...)
#     ↑        ↑         ↑
#     这些对 A, b, c 的导数很复杂

# JAX 自动处理
grad_fn = jax.grad(lambda p: compute_ec_with_params(p))
dA, db, dc, dy0 = grad_fn(params)  # 自动计算！
```

### 3. 多参数联合优化

```python
# 可以同时优化多个参数
params = LDAParameters(...)

# JAX 自动计算所有参数的梯度
gradients = jax.grad(loss_fn)(params)
# gradients.C_x, gradients.A, gradients.b, gradients.c, gradients.y0

# 一次性更新所有参数
params = update_all_params(params, gradients)
```

---

## 物理约束

优化参数时应注意物理约束：

```python
def loss_with_constraints(params: LDAParameters):
    # 物理约束
    if params.A < 0 or params.A > 0.1:
        return 1e10  # 大惩罚
    
    # 正常损失
    return regular_loss(params)
```

或使用约束优化器（如 optax 的 projected gradient）。

---

## 总结

**现在的设计优势：**

1. ✅ **所有参数都在 dataclass 中**
   - VWN 的 A, b, c, y0
   - PBE 的 mu, kappa
   - 等等

2. ✅ **通过 self.params 访问**
   - `A = self.params.A`
   - JAX 可以追踪计算图

3. ✅ **自动微分就绪**
   - 注册为 pytree 后立即可用
   - 无需手动推导梯度

4. ✅ **灵活优化**
   - 可以优化单个参数
   - 可以优化参数子集
   - 可以同时优化所有参数

**这就是为什么要把参数放到 dataclass 中！** 🎯

