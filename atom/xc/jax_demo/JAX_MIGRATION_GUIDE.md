# JAX Migration Guide for XC Functionals

## 为什么需要 JAX？

将 XC 泛函代码迁移到 JAX 可以实现：

1. **自动微分**：对泛函参数求导，用于 delta learning
2. **JIT 编译**：大幅提升性能（可达 10-100x）
3. **向量化**：使用 `vmap` 并行处理多个原子配置
4. **GPU 加速**：无缝切换到 GPU（如果可用）

## 核心设计原则

### 1. 纯函数（Pure Functions）

**当前设计（有状态）：**
```python
class LDA_SVWN(XCEvaluator):
    def __init__(self, derivative_matrix):
        self.derivative_matrix = derivative_matrix  # 状态
        self.alpha = -0.75  # 状态
    
    def compute_exchange_generic(self, rho, grad_rho=None, tau=None):
        # 依赖 self.alpha
        return self.alpha * rho**(4/3)
```

**JAX 友好设计（无状态）：**
```python
def compute_lda_exchange(rho: Array, alpha: float) -> Array:
    """纯函数：相同输入 → 相同输出，无副作用"""
    return alpha * rho**(4/3)

# JAX 可以自动求导
d_alpha = jax.grad(lambda a: jnp.sum(compute_lda_exchange(rho, a)))(alpha)
```

### 2. 显式参数传递

**关键改变：** 所有可能需要优化的系数都应该作为参数显式传递。

```python
@dataclass
class XCParams:
    # LDA
    alpha_x: float = -0.73855876638202234
    
    # PBE GGA
    mu: float = 0.2195149727645171
    kappa: float = 0.804
    
    # Delta learning corrections
    delta_alpha: float = 0.0  # 可训练！
    delta_mu: float = 0.0     # 可训练！

def compute_pbe_exchange(rho, grad_rho, params: XCParams):
    # 使用参数（而不是硬编码常量）
    mu = params.mu + params.delta_mu  # 加上 delta 修正
    # ... 计算
```

### 3. 数据结构兼容性

使用 `chex.dataclass` 或注册 pytree：

```python
import chex

@chex.dataclass
class GenericXCResult:
    v_generic: chex.Array
    e_generic: chex.Array
    de_dsigma: Optional[chex.Array] = None
    de_dtau: Optional[chex.Array] = None

# JAX 现在可以自动处理这个数据类
result = GenericXCResult(v=v_x, e=e_x)
grad_fn = jax.grad(lambda r: jnp.sum(r.e_generic))
gradients = grad_fn(result)  # ✓ 可以工作
```

## 迁移策略：双接口模式（推荐）

**优势：**
- 保持向后兼容
- 逐步迁移
- 同时支持 NumPy 和 JAX

**架构：**
```
Current Code (NumPy)  ←─────┐
                             │
                     [Adapter Layer]
                             │
                             ↓
                   Functional Core (JAX)
                             │
                             ↓
                   Delta Learning Pipeline
```

### 实现步骤

#### Step 1: 创建参数容器

```python
# delta/atomic_dft/xc/params.py
@dataclass
class XCParameters:
    functional_name: str
    
    # 基础参数
    base_params: Dict[str, float]
    
    # Delta learning 修正
    delta_params: Dict[str, float]
    
    def get_param(self, name: str) -> float:
        """获取参数（base + delta）"""
        base = self.base_params.get(name, 0.0)
        delta = self.delta_params.get(name, 0.0)
        return base + delta
```

#### Step 2: 提取核心计算为纯函数

```python
# delta/atomic_dft/xc/functionals/lda_functional.py
def lda_exchange_generic(
    rho: Array,
    alpha_x: float
) -> Tuple[Array, Array]:
    """
    纯函数：LDA 交换能
    
    可以被 JAX 自动微分：
    - 对 rho 求导 → 得到泛函导数
    - 对 alpha_x 求导 → 得到参数梯度（用于优化）
    """
    rho_43 = rho ** (4.0/3.0)
    rho_13 = rho ** (1.0/3.0)
    
    e_x = alpha_x * rho_43
    v_x = (4.0/3.0) * alpha_x * rho_13
    
    return v_x, e_x


def lda_exchange_generic_with_derivs(
    rho: Array,
    alpha_x: float
) -> GenericXCResult:
    """
    包装器：返回标准格式
    """
    v_x, e_x = lda_exchange_generic(rho, alpha_x)
    return GenericXCResult(
        v_generic=v_x,
        e_generic=e_x,
        de_dsigma=None,
        de_dtau=None
    )
```

#### Step 3: 创建适配器

```python
# delta/atomic_dft/xc/lda.py (保持 OOP 接口)
class LDA_SVWN(XCEvaluator):
    def __init__(self, derivative_matrix=None, params: Optional[XCParameters] = None):
        super().__init__(derivative_matrix)
        self.params = params or self._default_params()
    
    def _default_params(self) -> XCParameters:
        return XCParameters(
            functional_name='LDA_SVWN',
            base_params={'alpha_x': -0.73855876638202234},
            delta_params={}
        )
    
    def compute_exchange_generic(self, rho, grad_rho=None, tau=None) -> GenericXCResult:
        """OOP 接口 → 调用纯函数"""
        alpha_x = self.params.get_param('alpha_x')
        return lda_exchange_generic_with_derivs(rho, alpha_x)
```

#### Step 4: Delta Learning 流程

```python
# delta/atomic_dft/delta/train.py
import jax
import jax.numpy as jnp
import optax

def train_delta_functional(
    train_data: List[Tuple[Array, Array]],  # (rho, e_x_reference)
    initial_params: XCParameters,
    num_steps: int = 1000
):
    """
    训练 delta 修正参数
    """
    # 定义损失函数
    def loss_fn(params: XCParameters):
        total_loss = 0.0
        for rho, e_x_ref in train_data:
            # 计算 XC 能量
            alpha_x = params.get_param('alpha_x')
            result = lda_exchange_generic_with_derivs(rho, alpha_x)
            e_x_pred = result.e_generic
            
            # 计算误差
            error = jnp.sum((e_x_pred - e_x_ref) ** 2)
            total_loss += error
        
        return total_loss / len(train_data)
    
    # 设置优化器
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(initial_params)
    
    # 训练循环
    for step in range(num_steps):
        # 计算梯度
        loss, grads = jax.value_and_grad(loss_fn)(initial_params)
        
        # 更新参数
        updates, opt_state = optimizer.update(grads, opt_state)
        initial_params = optax.apply_updates(initial_params, updates)
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss:.6f}")
    
    return initial_params
```

## 具体修改建议

### 对于 `evaluator.py`

**当前设计已经很好！** 您的两阶段设计（generic → spherical）非常适合 JAX：

1. **保持接口不变**：`compute_xc()` 仍然是主入口
2. **内部调用纯函数**：实际计算委托给纯函数
3. **参数可选**：添加可选的 `params` 参数

```python
class XCEvaluator(ABC):
    def __init__(self, derivative_matrix=None, params: Optional[XCParameters] = None):
        self.derivative_matrix = derivative_matrix
        self.params = params  # 新增
    
    @abstractmethod
    def compute_exchange_generic(self, rho, grad_rho=None, tau=None) -> GenericXCResult:
        """
        子类实现：应该调用纯函数版本
        """
        pass
```

### 对于 `lda.py`, `gga_pbe.py` 等

**核心思想：** 分离计算逻辑和类接口

```python
# 文件结构
lda.py:
  - lda_exchange_kernel(rho, alpha)  # 纯函数，JAX 可导
  - lda_correlation_kernel(rho, ...)  # 纯函数
  - class LDA_SVWN(XCEvaluator)       # 适配器类
```

### 对于 GGA 和 meta-GGA

**关键点：** `de_dsigma` 和 `de_dtau` 可以用 JAX 自动计算！

```python
def pbe_exchange_energy_density(rho, sigma, mu, kappa):
    """只计算能量密度 ε_x(ρ, σ)"""
    # PBE 公式
    # ...
    return e_x

# 自动计算导数！
def pbe_exchange_with_derivatives(rho, sigma, mu, kappa):
    # ∂ε/∂ρ
    de_drho = jax.grad(lambda r: jnp.sum(pbe_exchange_energy_density(r, sigma, mu, kappa)))(rho)
    
    # ∂ε/∂σ
    de_dsigma = jax.grad(lambda s: jnp.sum(pbe_exchange_energy_density(rho, s, mu, kappa)))(sigma)
    
    # 能量密度
    e_x = pbe_exchange_energy_density(rho, sigma, mu, kappa)
    
    return GenericXCResult(
        v_generic=de_drho,
        e_generic=e_x,
        de_dsigma=de_dsigma,
        de_dtau=None
    )
```

**优势：** 您不需要手动推导 ∂ε/∂σ 的解析表达式！JAX 自动算！

## 性能考虑

### JIT 编译

```python
@jax.jit
def fast_xc_compute(rho, grad_rho, params):
    x_result = compute_exchange_generic_functional(rho, grad_rho, None, params)
    c_result = compute_correlation_generic_functional(rho, grad_rho, None, params)
    return x_result, c_result

# 第一次调用：编译（慢）
# 后续调用：非常快！
```

### 向量化（vmap）

```python
# 批量处理多个原子配置
rhos = jnp.array([rho1, rho2, rho3, ...])  # (batch, n_points)

# 自动向量化
batched_compute = jax.vmap(
    lambda r: compute_exchange_generic_functional(r, None, None, params)
)

results = batched_compute(rhos)  # 并行处理所有配置
```

## 实施时间线

### Phase 1: 准备（1-2 周）
- [ ] 安装 JAX 和依赖
- [ ] 创建 `XCParameters` 数据类
- [ ] 添加类型注解
- [ ] 单元测试覆盖现有功能

### Phase 2: 提取纯函数（2-3 周）
- [ ] LDA: 提取为纯函数
- [ ] GGA PBE: 提取为纯函数
- [ ] SCAN: 提取为纯函数
- [ ] 验证数值结果一致

### Phase 3: JAX 集成（1-2 周）
- [ ] 注册 pytrees
- [ ] 测试 `jax.grad`
- [ ] 测试 `jax.jit`
- [ ] 性能基准测试

### Phase 4: Delta Learning（2-4 周）
- [ ] 生成训练数据
- [ ] 实现损失函数
- [ ] 实现训练循环
- [ ] 验证物理约束

## 注意事项

### ⚠️ JAX 限制

1. **条件语句**：`if` 语句在 JIT 编译时会固化
   ```python
   # ❌ 不好
   def compute(x):
       if x > 0:  # 编译时固化！
           return x ** 2
   
   # ✓ 好
   def compute(x):
       return jnp.where(x > 0, x**2, 0.0)  # 动态选择
   ```

2. **In-place 修改**：JAX 数组不可变
   ```python
   # ❌ 不好
   v[mask] = 0.0  # 会报错！
   
   # ✓ 好
   v = jnp.where(mask, 0.0, v)
   ```

3. **NumPy 函数**：使用 `jax.numpy` 替代 `numpy`
   ```python
   # ❌ 不好
   import numpy as np
   result = np.sum(array)
   
   # ✓ 好
   import jax.numpy as jnp
   result = jnp.sum(array)
   ```

### ✓ 最佳实践

1. **渐进式迁移**：先迁移一个功能（如 LDA），完全测试后再继续
2. **保持 NumPy 后备**：如果 JAX 不可用，回退到 NumPy
3. **物理约束验证**：确保优化后的参数仍然满足物理要求
4. **正则化**：防止参数过度拟合训练数据

## 示例：完整的 LDA 迁移

参见 `evaluator_jax_compatible.py` 中的完整示例。

## 参考资源

- [JAX 官方文档](https://jax.readthedocs.io/)
- [JAX for Numerical Optimization](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)
- [Differentiable DFT (DQC library)](https://github.com/diffqc/dqc)
- [PySCF-forge JAX interface](https://github.com/pyscf/pyscf-forge)

