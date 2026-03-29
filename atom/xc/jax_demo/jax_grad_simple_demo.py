"""
Simple JAX Gradient Demo (ASCII only)

Shows what JAX gradients look like for different data structures.
"""

import numpy as np
from dataclasses import dataclass

# Check if JAX is available
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
    print("[OK] JAX is installed\n")
except ImportError:
    HAS_JAX = False
    print("[INFO] JAX not installed - showing conceptual examples only\n")


print("=" * 70)
print("What are JAX gradients?")
print("=" * 70)

if not HAS_JAX:
    print("""
Without JAX installed, here's the concept:

When you write:
    grad_fn = jax.grad(loss_function)
    gradients = grad_fn(params)

The 'gradients' object has THE SAME STRUCTURE as 'params':

1. If params is a scalar (float):
   gradients is a scalar: df/d(params)

2. If params is a dict:
   gradients is a dict with same keys:
   params = {'a': 2.0, 'b': 3.0}
   gradients = {'a': df/da, 'b': df/db}

3. If params is a dataclass:
   gradients is the SAME dataclass type:
   
   @dataclass
   class Params:
       alpha: float
       beta: float
   
   params = Params(alpha=2.0, beta=3.0)
   gradients = Params(alpha=df/dalpha, beta=df/dbeta)
   
   # You can access:
   print(gradients.alpha)  # This is df/dalpha
   print(gradients.beta)   # This is df/dbeta

4. Example usage (parameter 'update'):
   learning_rate = 0.01
   new_alpha = params.alpha - learning_rate * gradients.alpha
   new_beta = params.beta - learning_rate * gradients.beta

This is how Delta Learning works:
- You define trainable parameters (delta corrections)
- JAX computes gradients automatically
- You update parameters to minimize error
""")

else:
    # Actual JAX demo
    print("DEMO 1: Dictionary gradients")
    print("-" * 70)
    
    x = jnp.array([1.0, 2.0, 3.0])
    
    def f(params):
        a = params['a']
        b = params['b']
        return jnp.sum(a * x**2 + b * x)
    
    params = {'a': 2.0, 'b': 3.0}
    grad_fn = jax.grad(f)
    gradients = grad_fn(params)
    
    print(f"Function: f(a,b) = sum(a * x^2 + b * x)")
    print(f"x = {x}")
    print(f"params = {params}")
    print(f"\nGradients:")
    print(f"  type = {type(gradients)}")
    print(f"  value = {gradients}")
    print(f"\nExplanation:")
    print(f"  gradients['a'] = {gradients['a']} (this is df/da)")
    print(f"  gradients['b'] = {gradients['b']} (this is df/db)")
    
    print("\n" + "=" * 70)
    print("DEMO 2: Dataclass gradients (YOUR CASE!)")
    print("-" * 70)
    
    @dataclass
    class XCParams:
        alpha: float
        beta: float
    
    # Register as JAX pytree
    from jax import tree_util
    
    def flatten(p):
        return (p.alpha, p.beta), None
    
    def unflatten(aux, children):
        return XCParams(*children)
    
    tree_util.register_pytree_node(XCParams, flatten, unflatten)
    
    # Define function using dataclass
    def g(params: XCParams):
        return jnp.sum(params.alpha * x**2 + params.beta * x)
    
    params = XCParams(alpha=2.0, beta=3.0)
    grad_fn = jax.grad(g)
    gradients = grad_fn(params)
    
    print(f"Function: g(params) = sum(alpha * x^2 + beta * x)")
    print(f"params = {params}")
    print(f"\n*** KEY POINT: gradients has SAME TYPE as params ***")
    print(f"  type(gradients) = {type(gradients)}")
    print(f"  gradients = {gradients}")
    print(f"\nYou can access fields directly:")
    print(f"  gradients.alpha = {gradients.alpha}")
    print(f"  gradients.beta = {gradients.beta}")
    
    print("\n" + "=" * 70)
    print("DEMO 3: Parameter update (gradient descent)")
    print("-" * 70)
    
    learning_rate = 0.1
    new_alpha = params.alpha - learning_rate * gradients.alpha
    new_beta = params.beta - learning_rate * gradients.beta
    
    print(f"Original params: alpha={params.alpha}, beta={params.beta}")
    print(f"Gradients: alpha={gradients.alpha}, beta={gradients.beta}")
    print(f"Learning rate: {learning_rate}")
    print(f"\nUpdated params:")
    print(f"  alpha: {params.alpha} -> {new_alpha}")
    print(f"  beta: {params.beta} -> {new_beta}")
    
    new_params = XCParams(alpha=new_alpha, beta=new_beta)
    print(f"\nnew_params = {new_params}")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("""
Key Takeaways:

1. jax.grad(f) returns a FUNCTION that computes gradients
   
2. gradients = grad_fn(params)
   The gradients object has THE SAME STRUCTURE as params
   
3. For dataclass:
   - If params = MyClass(a=1.0, b=2.0)
   - Then gradients = MyClass(a=df/da, b=df/db)
   
4. Use gradients to update parameters:
   new_param = old_param - learning_rate * gradient
   
5. This enables Delta Learning:
   - Add trainable delta corrections to functional
   - Compute error vs. reference data
   - Use JAX to compute gradients automatically
   - Update parameters iteratively

For your XC functional code:
- Add XCParameters dataclass with delta corrections
- JAX will compute gradients w.r.t. ALL parameters
- Use gradient descent to optimize delta corrections
- This improves functional accuracy!
""")

