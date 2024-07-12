from functools import partial

import jax
import jax.numpy as jnp

def f_ref(x, y):
  return jnp.sin(x) * y

@partial(jax.custom_vjp, allow_jvp=True)
def f(x, y):
  return f_ref(x, y)

def f_fwd(x, y):
  return f_ref(x, y), (jnp.cos(x), jnp.sin(x), y)

def f_bwd(res, g):
  cos_x, sin_x, y = res
  return (cos_x * g * y, sin_x * g)

f.defvjp(f_fwd, f_bwd)

print(f(1.0, 2.0))
print(jax.grad(f)(1.0, 2.0))
