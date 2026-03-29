"""
JAX 自动微分详解与示例

这个文件展示了 JAX 如何计算梯度，特别是对于复杂数据结构（dataclass）的梯度。
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

# 模拟 JAX（如果没安装，用注释展示）
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit
    HAS_JAX = True
    print("[OK] JAX installed")
except ImportError:
    print("[INFO] JAX not installed. Showing conceptual examples.")
    HAS_JAX = False


# =============================================================================
# 例子 1: 简单标量参数的梯度
# =============================================================================

def example_1_simple_gradient():
    """
    最简单的例子：对单个标量参数求导
    """
    if not HAS_JAX:
        print("\n=== Example 1: Simple Gradient (requires JAX) ===")
        print("Conceptual explanation:")
        print("If f(x) = x^2, then df/dx = 2x")
        return
    
    print("\n=== Example 1: Simple Gradient ===")
    
    # 定义函数
    def f(x):
        return x ** 2
    
    # 创建梯度函数
    df_dx = jax.grad(f)
    
    # 计算 x=3.0 处的梯度
    x = 3.0
    gradient = df_dx(x)
    
    print(f"Function: f(x) = x^2")
    print(f"At x = {x}:")
    print(f"  f(x) = {f(x)}")
    print(f"  df/dx = {gradient}")  # 应该是 2*3 = 6.0
    print(f"  Expected: {2 * x}")
    
    return gradient


# =============================================================================
# 例子 2: 多个参数的梯度（字典）
# =============================================================================

def example_2_dict_gradient():
    """
    对字典类型的参数求导
    """
    if not HAS_JAX:
        print("\n=== Example 2: Dictionary Gradient (requires JAX) ===")
        print("Conceptual explanation:")
        print("params = {'a': 2.0, 'b': 3.0}")
        print("f(params) = a * x^2 + b * x")
        print("Gradient will be a dict with same structure:")
        print("  grad = {'a': df/da, 'b': df/db}")
        return
    
    print("\n=== Example 2: Dictionary Gradient ===")
    
    # 输入数据
    x = jnp.array([1.0, 2.0, 3.0])
    
    # 定义函数（使用字典参数）
    def f(params):
        a = params['a']
        b = params['b']
        return jnp.sum(a * x**2 + b * x)
    
    # 参数
    params = {'a': 2.0, 'b': 3.0}
    
    # 计算梯度
    grad_f = jax.grad(f)
    gradients = grad_f(params)
    
    print(f"Function: f(a, b) = sum(a * x^2 + b * x)")
    print(f"x = {x}")
    print(f"params = {params}")
    print(f"\nGradients (gradients 的类型和结构):")
    print(f"  type(gradients) = {type(gradients)}")
    print(f"  gradients = {gradients}")
    print(f"\n解释：")
    print(f"  df/da = sum(x^2) = {jnp.sum(x**2)}")
    print(f"  df/db = sum(x) = {jnp.sum(x)}")
    print(f"  计算出的梯度: a={gradients['a']}, b={gradients['b']}")
    
    return gradients


# =============================================================================
# 例子 3: Dataclass 的梯度（最重要！）
# =============================================================================

@dataclass
class SimpleParams:
    """简单的参数容器"""
    alpha: float
    beta: float

if HAS_JAX:
    # 注册为 JAX pytree（这样 JAX 才能处理它）
    from jax import tree_util
    
    def _simple_params_flatten(params):
        """告诉 JAX 如何展开 dataclass"""
        children = (params.alpha, params.beta)  # 可微分的部分
        aux_data = None  # 不可微分的元数据
        return children, aux_data
    
    def _simple_params_unflatten(aux_data, children):
        """告诉 JAX 如何重建 dataclass"""
        return SimpleParams(*children)
    
    tree_util.register_pytree_node(
        SimpleParams,
        _simple_params_flatten,
        _simple_params_unflatten
    )


def example_3_dataclass_gradient():
    """
    对 dataclass 求导（这就是您的 XCParameters 的情况！）
    """
    if not HAS_JAX:
        print("\n=== Example 3: Dataclass Gradient (requires JAX) ===")
        print("Conceptual explanation:")
        print("@dataclass")
        print("class Params:")
        print("    alpha: float")
        print("    beta: float")
        print("\nGradient will be SAME dataclass structure:")
        print("  gradients = Params(alpha=df/dalpha, beta=df/dbeta)")
        return
    
    print("\n=== Example 3: Dataclass Gradient (核心！) ===")
    
    # 输入数据
    x = jnp.array([1.0, 2.0, 3.0])
    
    # 定义函数（使用 dataclass 参数）
    def f(params: SimpleParams):
        return jnp.sum(params.alpha * x**2 + params.beta * x)
    
    # 参数
    params = SimpleParams(alpha=2.0, beta=3.0)
    
    # 计算梯度
    grad_f = jax.grad(f)
    gradients = grad_f(params)
    
    print(f"Function: f(params) = sum(alpha * x^2 + beta * x)")
    print(f"x = {x}")
    print(f"params = {params}")
    print(f"\n*** 关键：gradients 的类型和结构")
    print(f"  type(gradients) = {type(gradients)}")
    print(f"  gradients = {gradients}")
    print(f"\n解释：")
    print(f"  gradients 是一个 SimpleParams 实例！")
    print(f"  gradients.alpha = {gradients.alpha} (这是 df/dalpha)")
    print(f"  gradients.beta = {gradients.beta} (这是 df/dbeta)")
    print(f"\n验证：")
    print(f"  df/dalpha 应该等于 sum(x^2) = {jnp.sum(x**2)}")
    print(f"  df/dbeta 应该等于 sum(x) = {jnp.sum(x)}")
    
    return gradients


# =============================================================================
# 例子 4: 完整的 XC 参数梯度（您的实际用例）
# =============================================================================

@dataclass
class XCParameters:
    """完整的 XC 泛函参数（类似您代码中的）"""
    functional_name: str
    
    # 基础参数
    alpha_x: float = -0.73855876638202234  # LDA 交换
    
    # Delta learning 修正（这些可以被优化！）
    delta_alpha: float = 0.0
    
    # GGA 参数
    mu: float = 0.2195
    delta_mu: float = 0.0


if HAS_JAX:
    # 注册 XCParameters 为 pytree
    def _xc_params_flatten(params):
        # 只有数值字段参与微分，字符串不参与
        children = (
            params.alpha_x, 
            params.delta_alpha,
            params.mu,
            params.delta_mu
        )
        aux_data = {'functional_name': params.functional_name}  # 元数据
        return children, aux_data
    
    def _xc_params_unflatten(aux_data, children):
        return XCParameters(
            functional_name=aux_data['functional_name'],
            alpha_x=children[0],
            delta_alpha=children[1],
            mu=children[2],
            delta_mu=children[3]
        )
    
    tree_util.register_pytree_node(
        XCParameters,
        _xc_params_flatten,
        _xc_params_unflatten
    )


def lda_exchange_energy(rho, params: XCParameters):
    """LDA 交换能（使用 params）"""
    # 获取有效的 alpha（基础 + delta 修正）
    alpha_eff = params.alpha_x + params.delta_alpha
    
    # 计算交换能
    if HAS_JAX:
        e_x = jnp.sum(alpha_eff * rho ** (4.0/3.0))
    else:
        e_x = np.sum(alpha_eff * rho ** (4.0/3.0))
    
    return e_x


def example_4_xc_parameter_gradient():
    """
    完整示例：对 XC 参数求导（这就是 delta learning！）
    """
    if not HAS_JAX:
        print("\n=== Example 4: XC Parameter Gradient (requires JAX) ===")
        print("This shows how gradients enable delta learning.")
        print("\nConceptual flow:")
        print("1. Define XCParameters with delta corrections")
        print("2. Compute XC energy using these parameters")
        print("3. Compare with reference (high-level calculation)")
        print("4. Compute gradient of error w.r.t. delta parameters")
        print("5. Update parameters using gradient descent")
        return
    
    print("\n=== Example 4: XC Parameter Gradient (完整示例) ===")
    
    # 模拟电子密度
    rho = jnp.array([1.0, 2.0, 3.0, 4.0])
    
    # 初始参数（delta 修正为 0）
    params = XCParameters(
        functional_name='LDA_SVWN',
        alpha_x=-0.73855876638202234,
        delta_alpha=0.0  # 这个参数将被优化！
    )
    
    # 模拟"真实"能量（来自高精度计算）
    e_ref = -15.0
    
    # 定义损失函数（预测能量与参考能量的差异）
    def loss_fn(p: XCParameters):
        e_pred = lda_exchange_energy(rho, p)
        return (e_pred - e_ref) ** 2
    
    # 计算梯度
    grad_fn = jax.grad(loss_fn)
    gradients = grad_fn(params)
    
    print(f"密度: rho = {rho}")
    print(f"参考能量: e_ref = {e_ref}")
    print(f"\n当前参数:")
    print(f"  params.alpha_x = {params.alpha_x}")
    print(f"  params.delta_alpha = {params.delta_alpha}")
    
    e_pred = lda_exchange_energy(rho, params)
    print(f"\n预测能量: e_pred = {e_pred}")
    print(f"损失: L = (e_pred - e_ref)^2 = {loss_fn(params)}")
    
    print(f"\n*** 梯度（这就是 delta learning 的核心！）:")
    print(f"  type(gradients) = {type(gradients)}")
    print(f"  gradients = {gradients}")
    print(f"\n具体值:")
    print(f"  dL/d(alpha_x) = {gradients.alpha_x}")
    print(f"  dL/d(delta_alpha) = {gradients.delta_alpha}")
    print(f"  dL/d(mu) = {gradients.mu}")
    print(f"  dL/d(delta_mu) = {gradients.delta_mu}")
    
    print(f"\n使用梯度更新参数（梯度下降）:")
    learning_rate = 0.01
    new_delta_alpha = params.delta_alpha - learning_rate * gradients.delta_alpha
    print(f"  delta_alpha: {params.delta_alpha} → {new_delta_alpha}")
    
    # 验证：更新后损失应该减小
    new_params = XCParameters(
        functional_name='LDA_SVWN',
        alpha_x=params.alpha_x,
        delta_alpha=new_delta_alpha
    )
    new_loss = loss_fn(new_params)
    print(f"\n验证:")
    print(f"  旧损失: {loss_fn(params)}")
    print(f"  新损失: {new_loss}")
    print(f"  改善: {loss_fn(params) - new_loss}")
    
    return gradients


# =============================================================================
# 例子 5: 实际的训练循环
# =============================================================================

def example_5_training_loop():
    """
    完整的训练循环：迭代优化参数
    """
    if not HAS_JAX:
        print("\n=== Example 5: Training Loop (requires JAX) ===")
        print("This would show iterative parameter 'optimization'.")
        return
    
    print("\n=== Example 5: Training Loop (完整训练) ===")
    
    # 训练数据（多个密度配置）
    rhos = [
        jnp.array([1.0, 2.0, 3.0]),
        jnp.array([2.0, 3.0, 4.0]),
        jnp.array([1.5, 2.5, 3.5]),
    ]
    e_refs = [-10.0, -20.0, -15.0]  # 对应的参考能量
    
    # 初始参数
    params = XCParameters(
        functional_name='LDA_SVWN',
        alpha_x=-0.73855876638202234,
        delta_alpha=0.0  # 从 0 开始优化
    )
    
    # 定义总损失函数
    def total_loss(p: XCParameters):
        loss = 0.0
        for rho, e_ref in zip(rhos, e_refs):
            e_pred = lda_exchange_energy(rho, p)
            loss += (e_pred - e_ref) ** 2
        return loss / len(rhos)
    
    # 训练
    grad_fn = jax.grad(total_loss)
    learning_rate = 0.001
    
    print(f"初始参数: delta_alpha = {params.delta_alpha}")
    print(f"初始损失: {total_loss(params):.6f}\n")
    
    print("训练进度:")
    for step in range(10):
        # 计算梯度
        grads = grad_fn(params)
        
        # 更新参数（梯度下降）
        new_delta_alpha = params.delta_alpha - learning_rate * grads.delta_alpha
        params = XCParameters(
            functional_name='LDA_SVWN',
            alpha_x=params.alpha_x,
            delta_alpha=new_delta_alpha
        )
        
        # 打印进度
        loss = total_loss(params)
        if step % 2 == 0:
            print(f"  Step {step:2d}: delta_alpha = {params.delta_alpha:8.5f}, loss = {loss:.6f}")
    
    print(f"\n最终参数: delta_alpha = {params.delta_alpha}")
    print(f"最终损失: {total_loss(params):.6f}")
    print(f"损失减少: {100 * (1 - total_loss(params) / total_loss(XCParameters('LDA_SVWN'))):.1f}%")
    
    return params


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("JAX 自动微分详解")
    print("=" * 70)
    
    # 运行所有示例
    example_1_simple_gradient()
    example_2_dict_gradient()
    example_3_dataclass_gradient()
    example_4_xc_parameter_gradient()
    example_5_training_loop()
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
关键要点：

1. jax.grad(f) 返回一个新函数，该函数计算 f 的梯度

2. 对于简单标量：
   gradients 是一个标量（float）
   
3. 对于字典：
   gradients 是相同结构的字典
   gradient = {'param1': df/dparam1, 'param2': df/dparam2}

4. 对于 dataclass（您的情况！）：
   gradients 是相同类型的 dataclass
   如果 params = XCParameters(alpha=2.0, beta=3.0)
   那么 gradients = XCParameters(alpha=df/dalpha, beta=df/dbeta)
   
5. 使用梯度更新参数（梯度下降）：
   new_param = old_param - learning_rate * gradient
   
6. 这就是 delta learning 的核心机制！
   - 定义可训练的 delta 修正参数
   - 计算预测值与参考值的误差
   - 用 JAX 自动计算梯度
   - 迭代更新参数以减小误差
""")
    
    if not HAS_JAX:
        print("\n提示：安装 JAX 来运行实际代码:")
        print("  pip install jax jaxlib")

