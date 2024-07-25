import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.custom_ad import custom_ad

def f_ref(x, y):
  return jnp.sum(np.eye(1) * jnp.sin(x) * y)

@custom_ad
def f(x, y):
  print("prim")
  return f_ref(x, y)

@f.def_jvp
def f_jvp(primals, tangents):
  print("jvp")
  x, y = primals
  x_dot, y_dot = tangents
  primal_out = f_ref(x, y)
  tangent_out = jnp.cos(x) * x_dot * y + jnp.sin(x) * y_dot
  return primal_out, tangent_out

@f.def_fwd
def f_fwd(x, y):
  print("fwd")
  return f_ref(x, y), (jnp.cos(x), jnp.sin(x), y)

@f.def_bwd
def f_bwd(res, g):
  print("bwd")
  cos_x, sin_x, y = res
  return (cos_x * g * y, sin_x * g)

# f.def_vjp(f_fwd, f_bwd)

@f.def_lin
def f_lin(res, x_dot, y_dot):
  cos_x, sin_x, y = res
  return cos_x * x_dot * y + sin_x * y_dot

def g(x, y):
  return jax.lax.cond(True, f, lambda x, _: x, x, y)

# print(g(0.5, 1.3))
# print(f(0.5, 1.3))

print(jax.value_and_grad(g)(0.5, 1.3))

# # print(jax.jit(f)(0.5, 1.3))
# # print(jax.jvp(f, (0.5, 1.3), (1.0, 1.0)))
# # print(jax.jvp(f_ref, (0.5, 1.3), (1.0, 1.0)))
# # print(jax.value_and_grad(f)(0.5, 1.3))
# # print(jax.value_and_grad(f_ref)(0.5, 1.3))

# # print(jax.value_and_grad(f)(0.5, 1.3))
# # print(jax.value_and_grad(f_ref)(0.5, 1.3))

# # print(jax.jit(jax.value_and_grad(f))(0.5, 1.3))
# # print(jax.jit(jax.hessian(f))(0.5, 1.3))

# # # DCE
# # print(jax.jit(lambda *args: jax.jvp(f, *args)[0])((0.5, 1.3), (1.0, 1.0)))

# # Linearize
# y, lin = jax.linearize(f, 0.5, 1.3)
# print(jax.make_jaxpr(lin)(1.0, 1.0))
# # print(lin(1.0, 1.0))
# # lin = jax.jit(lin)
# # lin_trans = jax.linear_transpose(lin, 1.0, 1.0)
# # print(lin_trans(1.0))

# # y, lin = jax.linearize(f_ref, 0.5, 1.3)
# # lin = jax.jit(lin)
# # lin_trans = jax.linear_transpose(lin, 1.0, 1.0)
# # print(lin_trans(1.0))
