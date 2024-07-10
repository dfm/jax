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

from collections.abc import Callable

from jax import lax
from jax._src import api_util
from jax._src import core
from jax._src import linear_util as lu
from jax._src import source_info_util
from jax._src.util import unzip2, safe_map, safe_zip
from jax._src.tree_util import tree_flatten, tree_unflatten

map = safe_map
zip = safe_zip


def api_linearize(fun: Callable, *primals):
  api_util.check_callable(fun)
  fun = lu.wrap_init(fun)
  primals_in_flat, in_tree = tree_flatten(primals)
  flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
  primals_out_flat, linear_flat = linearize(flat_fun).call_wrapped(primals_in_flat, map(Nothing, primals_flat))
  primals_out = tree_unflatten(out_tree, primals_out_flat)


def linearize(fun: lu.WrappedFun) -> lu.WrappedFun:
  return linearize_outer(linearize_inner(fun))


@lu.transformation
def linearize_outer(primals, linears):
  parent_trace = core.find_cur_trace()
  tag = LinarizeTag()
  with source_info_util.transform_name_stack("linearize"):
    out_primals, out_linears = yield (parent_trace, tag, primals, linears), {}
  yield out_primals, out_linears


@lu.transformation
def linearize_inner(parent_trace, tag, primals, linears):
  trace = LinearizeTrace(parent_trace, tag)
  in_tracers = [
    LinearizeTracer(trace, primal, linear)
    if not isinstance(linear, Nothing)
    else primal
    for primal, linear in zip(primals, linears)
  ]
  with core.set_current_trace(trace):
    ans = yield in_tracers, {}
  yield unzip2(map(trace.to_primal_linear_pair, ans))


class Nothing:
  pass


class LinarizeTag:
  pass


class LinearizeTrace(core.Trace):
  def __init__(self, parent_trace, tag):
    self.parent_trace = parent_trace
    self.tag = tag

  def to_primal_linear_pair(self, tracer):
    if isinstance(tracer, LinearizeTracer) and tracer._trace.tag == self.tag:
      return tracer.primal, tracer.linear
    else:
      return tracer, Nothing()

  def process_primitive(self, primitive, tracers, params):
    primals_in, linears_in = unzip2(map(self.to_primal_linear_pair, tracers))
    rule = primitive_linearize_rules.get(primitive)
    if not rule:
      msg = f"Linearization rule for '{primitive}' not implemented"
      raise NotImplementedError(msg)
    with core.set_current_trace(self.parent_trace):
      primals_out, linears_out = rule(primals_in, linears_in, **params)
    if primitive.multiple_results:
      return [LinearizeTracer(self, x, lin) for x, lin in zip(primals_out, linears_out)]
    else:
      return LinearizeTracer(self, primals_out, linears_out)


class LinearizeTracer(core.Tracer):
  __slots__ = ["primal", "linear"]

  def __init__(self, trace, primal, linear):
    self._trace = trace
    self.primal = primal
    self.linear = linear

  @property
  def aval(self):
    return core.get_aval(self.primal)

  def full_lower(self):
    if isinstance(self.linear, Nothing):
      return core.full_lower(self.primal)
    else:
      return self


primitive_linearize_rules: dict[core.Primitive, Callable] = {}


def linearize_sin(primals, linears):
  primal, = primals
  linear, = linears
  return lax.sin(primal), lambda: lax.cos(primal) * linear()

primitive_linearize_rules[lax.sin_p] = linearize_sin


if __name__ == "__main__":
  from jax import lax
  import jax.numpy as jnp

  def fun(x):
    return lax.sin(x)

  lin = linearize(lu.wrap_init(fun))
  print(lin.call_wrapped((jnp.array(1.0),), (Nothing,)))
