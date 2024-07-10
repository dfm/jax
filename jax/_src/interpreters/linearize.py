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
from jax._src.util import unzip2, safe_map, safe_zip, toposort
from jax._src.tree_util import tree_flatten, tree_unflatten, Partial

map = safe_map
zip = safe_zip


def api_linearize(fun: Callable, *primals):
  api_util.check_callable(fun)
  fun = lu.wrap_init(fun)
  primals_in_flat, in_tree = tree_flatten(primals)
  flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
  linears_in_flat = [LinearNode(None, []) for _ in primals_in_flat]
  primals_out_flat, linears_out_flat = linearize(flat_fun).call_wrapped(
    primals_in_flat, linears_in_flat
  )

  def linear(*tangents):
    tangents_in_flat, tangents_tree = tree_flatten(tangents)
    assert tangents_tree == in_tree, "TODO"
    env = dict(zip(linears_in_flat, tangents_in_flat))
    for node in toposort(linears_out_flat):
      if node in env:
        continue
      assert node.linear, "TODO"
      result = node.linear(*map(env.get, node.parents))
      env[node] = result
    tangents_out_flat = map(env.get, linears_out_flat)
    return tree_unflatten(out_tree(), tangents_out_flat)

  primals_out = tree_unflatten(out_tree(), primals_out_flat)
  return primals_out, Partial(linear)


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
      LinearizeTracer(trace, primal, linear) if linear else primal
      for primal, linear in zip(primals, linears)
  ]
  with core.set_current_trace(trace):
    ans = yield in_tracers, {}
  yield unzip2(map(trace.to_primal_linear_pair, ans))


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
      return tracer, None

  def process_primitive(self, primitive, tracers, params):
    primals_in, linears_in = unzip2(map(self.to_primal_linear_pair, tracers))
    rule = primitive_linearize_rules.get(primitive)
    if not rule:
      return self.default_process_primitive(primitive, tracers, params)
    with core.set_current_trace(self.parent_trace):
      primals_out, linears_out = rule(*primals_in, **params)
    if primitive.multiple_results:
      return [
        LinearizeTracer(self, x, LinearNode(lin, linears_in))
        for x, lin in zip(primals_out, linears_out)
      ]
    else:
      return LinearizeTracer(self, primals_out, LinearNode(linears_out, linears_in))

  def default_process_primitive(self, primitive, tracers, params):
    raise NotImplementedError("TODO: Implement default linearize rule using JVP+PE")


class LinearNode:
  __slots__ = ["linear", "parents"]

  def __init__(self, linear, parents):
    self.linear = linear
    self.parents = parents


class LinearizeTracer(core.Tracer):
  __slots__ = ["primal", "linear"]

  def __init__(self, trace, primal, linear):
    self._trace = trace
    self.primal = primal
    self.linear = linear

  @property
  def aval(self):
    return core.get_aval(self.primal)


primitive_linearize_rules: dict[core.Primitive, Callable] = {}

def linearize_sin(x):
  return lax.sin(x), lambda t: lax.cos(x) * t
primitive_linearize_rules[lax.sin_p] = linearize_sin

def linearize_add(x, y):
  return lax.add(x, y), lambda xt, yt: xt + yt
primitive_linearize_rules[lax.add_p] = linearize_add


if __name__ == "__main__":
  import jax
  from jax import lax
  import jax.numpy as jnp

  def fun(x):
    return lax.add(lax.sin(x), x)

  y, lin = api_linearize(fun, jnp.array(1.0))
  print(y)
  print(lin(jnp.array(1.0)))
  print(jax.jvp(fun, (jnp.array(1.0),), (jnp.array(1.0),)))
