# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from functools import wraps

import numpy as np
from absl.testing import absltest, parameterized

import jax
import jax.numpy as jnp
from jax import lax
from jax._src import test_util as jtu
from jax.experimental.custom_ad import custom_ad


def count_calls(f):
  @wraps(f)
  def wrapped(*args, **kwargs):
    print(f"called {f.__name__}")
    wrapped.calls += 1
    return f(*args, **kwargs)

  wrapped.calls = 0
  return wrapped


class CustomADTest(jtu.JaxTestCase):
  def test_jvp_basic(self):
    @custom_ad
    def f(x):
      return jnp.sin(x)
    @f.def_jvp
    def _(primals, tangents):
      x, = primals
      g, = tangents
      return f(x), 2 * jnp.cos(x) * g

    x = 3.
    self.assertAllClose(f(x), jnp.sin(x))
    self.assertAllClose(jax.jvp(f, (x,), (1.,)),
                        (jnp.sin(x), 2 * jnp.cos(x)))
    self.assertAllClose(jax.grad(f)(x), 2 * jnp.cos(x))

  def test_jvp_invariance(self):
    @custom_ad
    def f(x):
      return jnp.cos(2 * x) / 2.
    @f.def_jvp
    def _(primals, tangents):
      x, = primals
      g, = tangents
      return (f(x), 3 * g)
    def f2(x):
      y, _ = jax.jvp(f, (x,), (x,))
      return y
    def f3(x):
      y, _ = jax.jvp(f2, (x,), (x,))
      return y
    x = 1.
    self.assertAllClose(jax.jvp(f, (x,), (x,)),
                        jax.jvp(f2, (x,), (x,)),
                        check_dtypes=False)
    self.assertAllClose(jax.jvp(f, (x,), (x,)),
                        jax.jvp(f3, (x,), (x,)),
                        check_dtypes=False)

  # def test_jvp_vmap(self):
  #   @jax.custom_jvp
  #   def f(x):
  #     assert jnp.ndim(x) == 0
  #     return jnp.sin(x)
  #   def f_jvp(primals, tangents):
  #     x, = primals
  #     g, = tangents
  #     assert jnp.ndim(x) == jnp.ndim(g) == 0
  #     return f(x), 2 * jnp.cos(x) * g
  #   f.defjvp(f_jvp)

  #   x = jnp.arange(3.)
  #   xx = jnp.arange(6.).reshape(2, 3)

  #   # vmap of f
  #   self.assertAllClose(api.vmap(f)(x), jnp.sin(x))
  #   self.assertAllClose(api.vmap(api.vmap(f))(xx), jnp.sin(xx))

  #   # vmap of jvp of f
  #   self.assertAllClose(api.vmap(lambda x: api.jvp(f, (x,), (x,)))(x),
  #                       (jnp.sin(x), 2 * jnp.cos(x) * x))
  #   self.assertAllClose(api.vmap(api.vmap(lambda x: api.jvp(f, (x,), (x,))))(xx),
  #                       (jnp.sin(xx), 2 * jnp.cos(xx) * xx))

  #   # jvp of vmap of f
  #   self.assertAllClose(api.jvp(api.vmap(f), (x,), (x,)),
  #                       (jnp.sin(x), 2 * jnp.cos(x) * x))
  #   self.assertAllClose(api.jvp(api.vmap(api.vmap(f)), (xx,), (xx,)),
  #                       (jnp.sin(xx), 2 * jnp.cos(xx) * xx))

  #   # vmap of jvp of vmap of f
  #   self.assertAllClose(api.vmap(lambda x: api.jvp(api.vmap(f), (x,), (x,)))(xx),
  #                       (jnp.sin(xx), 2 * jnp.cos(xx) * xx))

  def test_jvp_jit(self):
    @custom_ad
    def f(x):
      return jnp.sin(x)
    @f.def_jvp
    def _(primals, tangents):
      x, = primals
      g, = tangents
      return f(x), 2 * jnp.cos(x) * g

    x = 3.
    self.assertAllClose(jax.jit(f)(x), jnp.sin(x))
    self.assertAllClose(jax.jit(jax.jit(f))(x), jnp.sin(x))
    self.assertAllClose(jax.jit(lambda x: jax.jvp(f, (x,), (x,)))(x),
                        (jnp.sin(x), 2 * jnp.cos(x) * x),
                        check_dtypes=False)
    self.assertAllClose(jax.jvp(jax.jit(f), (x,), (x,)),
                        (jnp.sin(x), 2 * jnp.cos(x) * x),
                        check_dtypes=False)

  def test_jvp_pytrees(self):
    @custom_ad
    def f(x):
      return {'b': jnp.sin(x['a'])}
    @f.def_jvp
    def _(primals, tangents):
      x, = primals
      g, = tangents
      return f(x), {'b': 2 * jnp.cos(x['a']) * g['a']}
    x = {'a': 3.}
    self.assertAllClose(f(x)['b'], jnp.sin(x['a']))
    self.assertAllClose(jax.jvp(f, (x,), (x,)),
                        ({'b': jnp.sin(x['a'])},
                         {'b': 2 * jnp.cos(x['a']) * x['a']}),
                        check_dtypes=False)

  def test_jvp_kwargs(self):
    @custom_ad
    def my_fun(x, y, c=1.):
      return c * (x + y)
    @my_fun.def_jvp
    def my_jvp(primals, tangents):
      x, y, c = primals
      t_x, t_y, t_c = tangents
      return my_fun(x, y, c), t_c
    f = lambda x, y: jnp.square(my_fun(x, y, c=2.)).sum()
    f(10., 5.)  # doesn't crash
    jax.jvp(f, (10., 5.), (1., 1.))  # doesn't crash

  def test_jvp_initial_style(self):
    @custom_ad
    def f(x):
      return 3 * x
    @f.def_jvp
    def _(primals, tangents):
      x, = primals
      g, = tangents
      return f(x), 2 * g

    def foo(x):
      out, _  = lax.scan(lambda c, _: (f(c), None), x, None, length=1)
      return out

    ans = jax.grad(foo)(3.)
    expected = 2.
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = jax.grad(jax.jit(foo))(3.)
    expected = 2.
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = jax.jit(jax.grad(foo))(3.)
    expected = 2.
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = jax.grad(jax.grad(foo))(3.)
    expected = 0.
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = jax.grad(jax.grad(jax.jit(foo)))(3.)
    expected = 0.
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = jax.grad(jax.jit(jax.grad(foo)))(3.)
    expected = 0.
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = jax.jit(jax.grad(jax.grad(foo)))(3.)
    expected = 0.
    self.assertAllClose(ans, expected, check_dtypes=False)

  # TODO(dfm): CustomJVPTest.test_initial_style_vmap
  # TODO(dfm): CustomJVPTest.test_initial_style_vmap_with_collective

  def test_jvp_closed_over_tracers_error_message(self):
    def f(x):
      @custom_ad
      def g(y):
        return x + y
      @g.def_jvp
      def _(primals, tangents):
        return g(x), 2 * primals[0]
      return g(1.)

    # TODO(dfm): Better error message.
    jax.grad(f)(3.)

    # self.assertRaises(ad.CustomJVPException, lambda: api.jvp(f, (3.,), (1.,)))
    # self.assertRaises(ad.CustomJVPException, lambda: api.grad(f)(3.))

  @parameterized.parameters([(False,), (True,)])
  def test_jvp(self, with_jit: bool):
    wrap = jax.jit if with_jit else lambda f: f

    @custom_ad
    def fun(x, y):
      # Include the identity matrix to make sure that we can handle consts.
      return jnp.sum(np.eye(1) * jnp.sin(x) * y)

    primals = (0.5, 1.3)
    tangents = (1.2, 0.7)
    self.assertAllClose(jax.jvp(wrap(fun), primals, tangents),
                        jax.jvp(fun.fun, primals, tangents))

    @fun.def_jvp
    def _(primals, tangents):
      x, y = primals
      x_dot, y_dot = tangents
      primal_out = fun(x, y)
      tangent_out = jnp.cos(x) * x_dot * y + jnp.sin(x) * y_dot
      return primal_out, tangent_out

    self.assertAllClose(jax.jvp(wrap(fun), primals, tangents),
                        jax.jvp(fun.fun, primals, tangents))

    @fun.def_fwd
    def _(x, y):
      return fun(x, y), (jnp.cos(x), jnp.sin(x), y)

    @fun.def_bwd
    def _(res, g):
      cos_x, sin_x, y = res
      return (cos_x * g * y, sin_x * g)

    self.assertAllClose(jax.jvp(wrap(fun), primals, tangents),
                        jax.jvp(fun.fun, primals, tangents))


  # def build_fun(self, with_consts: bool = False, with_cond: bool = False):
  #   @custom_ad
  #   @count_calls
  #   def f(x, y):
  #     if with_consts:
  #       return jnp.sum(np.eye(1) * jnp.sin(x) * y)
  #     else:
  #       return jnp.sin(x) * y

  #   @f.def_jvp
  #   @count_calls
  #   def f_jvp(primals, tangents):
  #     x, y = primals
  #     x_dot, y_dot = tangents
  #     primal_out = f(x, y)
  #     tangent_out = jnp.cos(x) * x_dot * y + jnp.sin(x) * y_dot
  #     return primal_out, tangent_out

  #   @f.def_fwd
  #   @count_calls
  #   def f_fwd(x, y):
  #     return f(x, y), (jnp.cos(x), jnp.sin(x), y)

  #   @f.def_bwd
  #   @count_calls
  #   def f_bwd(res, g):
  #     cos_x, sin_x, y = res
  #     return (cos_x * g * y, sin_x * g)

  #   @f.def_lin
  #   @count_calls
  #   def f_lin(res, x_dot, y_dot):
  #     cos_x, sin_x, y = res
  #     return cos_x * x_dot * y + sin_x * y_dot

  #   if with_cond:
  #     f_ = f

  #     @wraps(f_)
  #     def f(x, y):
  #       return lax.cond(True, f_, lambda x, _: x, x, y)

  #   primals = (0.5, 1.3)
  #   tangents = (1.2, 0.7)
  #   cotangents = (0.7,)
  #   return f, primals, tangents, cotangents

  # @parameterized.parameters([
  #   dict(with_consts=with_consts, with_cond=with_cond)
  #   for with_consts in [True, False]
  #   for with_cond in [True, False]
  # ])
  # def test_custom_ad_primal(self, with_consts: bool, with_cond: bool):
  #   f, primals, *_ = self.build_fun(with_consts=with_consts, with_cond=with_cond)
  #   x = f(*primals)
  #   assert f.fun.calls == 1
  #   self.assertAllClose(x, f.fun(*primals))

  # @parameterized.parameters([
  #   dict(with_consts=with_consts, with_cond=with_cond)
  #   for with_consts in [True, False]
  #   for with_cond in [True, False]
  # ])
  # def test_custom_ad_jvp(self, with_consts: bool, with_cond: bool):
  #   f, primals, tangents, _ = self.build_fun(
  #     with_consts=with_consts, with_cond=with_cond
  #   )
  #   self.assertAllClose(
  #     jax.jvp(f, primals, tangents), jax.jvp(f.fun, primals, tangents)
  #   )
  #   assert f.jvp.calls == 1
  #   assert f.fwd.calls == 0

  # @parameterized.parameters([
  #   dict(with_consts=with_consts, with_cond=with_cond)
  #   for with_consts in [True, False]
  #   for with_cond in [True, False]
  # ])
  # def test_custom_ad_vjp(self, with_consts: bool, with_cond: bool):
  #   f, primals, _, cotangents = self.build_fun(
  #     with_consts=with_consts, with_cond=with_cond
  #   )
  #   x, vjp = jax.vjp(f, *primals)
  #   x_, vjp_ = jax.vjp(f.fun, *primals)
  #   self.assertAllClose(x, x_)
  #   self.assertAllClose(vjp(*cotangents), vjp_(*cotangents))
  #   assert f.jvp.calls == 0
  #   assert f.fwd.calls == 1
  #   assert f.bwd.calls == 1
  #   assert f.lin.calls == 0

  # @parameterized.parameters([
  #   dict(with_consts=with_consts, with_cond=with_cond)
  #   for with_consts in [True, False]
  #   for with_cond in [True, False]
  # ])
  # def test_custom_ad_lin(self, with_consts: bool, with_cond: bool):
  #   f, primals, tangents, _ = self.build_fun(
  #     with_consts=with_consts, with_cond=with_cond
  #   )
  #   x, lin = jax.linearize(f, *primals)
  #   x_, lin_ = jax.linearize(f.fun, *primals)
  #   self.assertAllClose(x, x_)
  #   self.assertAllClose(lin(*tangents), lin_(*tangents))
  #   assert f.jvp.calls == 0
  #   assert f.fwd.calls == 1
  #   assert f.bwd.calls == 0
  #   assert f.lin.calls == 1

  # @parameterized.parameters([
  #   dict(with_consts=with_consts, with_cond=with_cond)
  #   for with_consts in [True, False]
  #   for with_cond in [True, False]
  # ])
  # def test_custom_ad_transpose(self, with_consts: bool, with_cond: bool):
  #   f, primals, _, cotangents = self.build_fun(
  #     with_consts=with_consts, with_cond=with_cond
  #   )
  #   _, lin = jax.linearize(f, *primals)
  #   trans = jax.linear_transpose(lin, *primals)
  #   _, lin_ = jax.linearize(f.fun, *primals)
  #   trans_ = jax.linear_transpose(lin_, *primals)
  #   self.assertAllClose(trans(*cotangents), trans_(*cotangents))
  #   assert f.jvp.calls == 0
  #   assert f.fwd.calls == 1
  #   assert f.bwd.calls == 1
  #   assert f.lin.calls == 0


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
