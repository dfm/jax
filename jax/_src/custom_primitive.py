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

from functools import partial, update_wrapper

from jax._src import api_util
from jax._src import core
from jax._src import linear_util as lu
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.interpreters.batching import not_mapped
from jax._src.lax.lax import tie_p
from jax._src.tree_util import tree_flatten, tree_unflatten
from jax._src.util import moveaxis, safe_map, safe_zip, split_list

map = safe_map
zip = safe_zip


class CustomPrimitive(core.Primitive):
  multiple_results = True

  def __init__(self, name: str, spec):
    super().__init__(name)
    update_wrapper(self, spec)
    self.spec = spec

  def __call__(self, *args, **kwargs):
    fun = lu.wrap_init(lambda *x: self.spec.impl(*x, **kwargs))
    args_flat, in_tree = tree_flatten(args)

    fun_flat, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
    avals_in = [core.raise_to_shaped(core.get_aval(x)) for x in args_flat]
    debug = pe.debug_info(fun, in_tree, out_tree, False, self.name)
    jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(fun_flat, avals_in, debug)
    call_jaxpr = core.ClosedJaxpr(pe.convert_constvars_jaxpr(jaxpr), ())
    out_tree = out_tree()

    jvp = getattr(self.spec, "jvp", None)
    if jvp is not None:
      jvp = flatten_jvp(lu.wrap_init(jvp), len(consts), in_tree, out_tree,
                        kwargs)
      jvp = Rule("jvp", jvp)

    vjp_fwd = getattr(self.spec, "vjp_fwd", None)
    vjp_bwd = None
    res_tree = None
    if vjp_fwd is not None:
      vjp_fwd, res_tree = flatten_vjp_fwd(
          lu.wrap_init(vjp_fwd), len(consts), in_tree, out_tree, kwargs,
          call_jaxpr.out_avals)
      vjp_fwd = Rule("vjp_fwd", vjp_fwd)

      vjp_bwd = self.spec.vjp_bwd
      vjp_bwd = flatten_vjp_bwd(lu.wrap_init(vjp_bwd), len(consts), in_tree,
                                out_tree, kwargs, res_tree)
      vjp_bwd = Rule("vjp_bwd", vjp_bwd)

    transpose = getattr(self.spec, "transpose", None)
    if transpose is not None:
      transpose = flatten_transpose(lu.wrap_init(transpose), len(consts),
                                    in_tree, out_tree, kwargs)
      transpose = Rule("transpose", transpose)

    vmap = getattr(self.spec, "vmap", None)
    if vmap is not None:
      vmap = flatten_vmap(lu.wrap_init(vmap), len(consts), in_tree, out_tree,
                          kwargs)
      vmap = Rule("vmap", vmap)

    out_flat = self.bind(*consts, *args_flat, call_jaxpr=call_jaxpr,
                         jvp=jvp, vjp_fwd=vjp_fwd, vjp_bwd=vjp_bwd,
                         res_tree=res_tree, transpose=transpose, vmap=vmap)

    return tree_unflatten(out_tree, out_flat)


class Rule:
  __slots__ = ["rule_type", "fun"]
  def __init__(self, rule_type: str, fun: lu.WrappedFun):
    self.rule_type = rule_type
    self.fun = fun

  def __repr__(self) -> str:
    return f"<CustomPrimitive {self.rule_type} rule>"

  def call_wrapped(self, *args, **kwargs):
    return self.fun.call_wrapped(*args, **kwargs)


def build_custom_primitive(spec, *, name: str | None = None) -> CustomPrimitive:
  if name is None:
    name = getattr(spec, "__name__", str(spec))

  if not hasattr(spec, "impl"):
    raise TypeError("A custom primitive requires an `impl`.")

  if hasattr(spec, "vjp_fwd") ^ hasattr(spec, "vjp_bwd"):
    raise TypeError(
        "A custom primitive must implement both or neither of the methods "
        "`vjp_fwd` and `vjp_bwd`.")

  if hasattr(spec, "jvp") and hasattr(spec, "vjp_fwd"):
    raise TypeError(
        "A custom primitive must not implement both `jvp` and `vjp` methods.")

  prim = CustomPrimitive(name, spec)
  prim.def_impl(custom_primitive_impl)
  prim.def_effectful_abstract_eval(custom_primitive_abstract_eval)
  mlir.register_lowering(prim, custom_primitive_lowering)
  ad.primitive_jvps[prim] = partial(custom_primitive_jvp, name)
  ad.primitive_transposes[prim] = partial(custom_primitive_transpose, name)
  batching.primitive_batchers[prim] = partial(custom_primitive_vmap, name)

  return prim


def custom_primitive_impl(*args, call_jaxpr: core.ClosedJaxpr, **_):
  return core.jaxpr_as_fun(call_jaxpr)(*args)


def custom_primitive_abstract_eval(*args, call_jaxpr: core.ClosedJaxpr, **_):
  del args
  # TODO(dfm): Check for allowed effects
  return call_jaxpr.out_avals, call_jaxpr.effects


def custom_primitive_lowering(ctx, *args, call_jaxpr: core.ClosedJaxpr, **_):
  args_ = map(mlir.wrap_singleton_ir_values, args)
  consts = mlir._ir_consts(call_jaxpr.consts)
  out, tokens = mlir.jaxpr_subcomp(ctx.module_context, call_jaxpr.jaxpr,
                                   ctx.name_stack, ctx.tokens_in, consts,
                                   *args_, dim_var_values=ctx.dim_var_values)
  ctx.set_tokens_out(tokens)
  return out


@lu.transformation
def flatten_jvp(num_consts, in_tree, out_tree, kwargs, primals, tangents):
  _, primals = split_list(primals, [num_consts])
  const_tangents, tangents = split_list(tangents, [num_consts])
  assert all(isinstance(t, ad.Zero) for t in const_tangents)
  py_primals = tree_unflatten(in_tree, primals)
  py_tangents = tree_unflatten(in_tree, tangents)
  py_primals_out, py_tangents_out = yield (py_primals, py_tangents), kwargs
  primals_out, out_tree1 = tree_flatten(py_primals_out)
  tangents_out, out_tree2 = tree_flatten(py_tangents_out)
  assert out_tree1 == out_tree, "todo error"
  assert out_tree2 == out_tree, "todo error"
  yield primals_out, tangents_out


@lu.transformation_with_aux
def flatten_vjp_fwd(num_consts, in_tree, out_tree, kwargs, out_avals, *args):
  _, args = split_list(args, [num_consts])
  py_args = tree_unflatten(in_tree, args)
  py_out, py_res = yield py_args, kwargs
  out, out_tree_ = tree_flatten(py_out)
  res, res_tree = tree_flatten(py_res)
  assert out_tree_ == out_tree, "todo error"
  out_avals_ = [core.raise_to_shaped(core.get_aval(x)) for x in out]
  assert all(map(core.typematch, out_avals_, out_avals)), "todo error"
  yield (*res, *out), res_tree


@lu.transformation
def flatten_vjp_bwd(num_consts, in_tree, out_tree, kwargs, res_tree_thunk,
                    *args):
  res_tree = res_tree_thunk()
  assert len(args) == res_tree.num_leaves + out_tree.num_leaves
  res, cts_out = split_list(args, [res_tree.num_leaves])
  py_res = tree_unflatten(res_tree, res)
  py_cts_out = tree_unflatten(out_tree, cts_out)
  py_cts_in = yield (py_res, py_cts_out), kwargs
  cts_in, in_tree_ = tree_flatten(py_cts_in, is_leaf=lambda x: x is None)
  assert in_tree_ == in_tree, "todo error"
  # TODO(dfm): Check types of cts_in match avals_in
  yield cts_in


def custom_primitive_jvp(name: str, primals, tangents,
                         call_jaxpr: core.ClosedJaxpr, jvp: Rule | None,
                         vjp_fwd: Rule | None, vjp_bwd: Rule | None, res_tree,
                         **_):
  del call_jaxpr

  if vjp_fwd is not None:
    assert vjp_bwd is not None
    fwd_in = [core.full_lower(x) for x in primals]
    res_and_primal = vjp_fwd.call_wrapped(*fwd_in)
    num_res = res_tree().num_leaves
    res, primals_out = split_list(res_and_primal, [num_res])
    avals_out = [core.raise_to_shaped(core.get_aval(x)) for x in primals_out]
    tangents_in = map(ad.instantiate_zeros, tangents)
    tangents_out = ad.custom_lin_p.bind(
        *res, *tangents_in, num_res=num_res, bwd=vjp_bwd.call_wrapped,
        out_avals=avals_out, symbolic_zeros=False)
    tangents_out = map(tie_p.bind, primals_out, tangents_out)
    tangents_out = map(ad.recast_to_float0, primals_out, tangents_out)
    return primals_out, tangents_out

  if jvp is None:
    raise NotImplementedError(
        f"'jvp' not implemented for custom primitive '{name}'")
  return jvp.call_wrapped(primals, tangents)


@lu.transformation
def flatten_transpose(num_consts, in_tree, out_tree, kwargs, cts_in, *args):
  _, args = split_list(args, [num_consts])
  py_args = tree_unflatten(in_tree, args)
  py_cts_in = tree_unflatten(out_tree, cts_in)
  py_cts_out = yield (py_cts_in, *py_args), kwargs
  cts_out, cts_out_tree = tree_flatten(py_cts_out, is_leaf=lambda x: x is None)
  assert cts_out_tree == in_tree, "todo error"
  yield [None] * num_consts + cts_out


def custom_primitive_transpose(name: str, cts_in, *args,
                               call_jaxpr: core.ClosedJaxpr, jvp: Rule | None,
                               vjp_fwd: Rule | None, vjp_bwd: Rule | None,
                               res_tree, transpose: Rule | None, **_):
  del call_jaxpr, jvp, vjp_fwd, vjp_bwd
  if transpose is None:
    raise NotImplementedError(
        f"'transpose' not implemented for custom primitive '{name}'")
  return transpose.call_wrapped(cts_in, *args)


@lu.transformation
def flatten_vmap(num_consts, in_tree, out_tree, kwargs, args, dims):
  _, args = split_list(args, [num_consts])
  const_dims, dims = split_list(dims, [num_consts])
  assert all(d is not_mapped for d in const_dims)

  axis_size, = {x.shape[d] for x, d in zip(args, dims) if d is not not_mapped}
  args = [x if d is not_mapped else moveaxis(x, d, 0) for x, d in zip(args, dims)]

  py_args = tree_unflatten(in_tree, args)
  args_batched = tree_unflatten(in_tree, [d is not not_mapped for d in dims])
  py_out, out_batched = yield (axis_size, args_batched, *py_args), kwargs
  out, out_tree1 = tree_flatten(py_out)
  batched, out_tree2 = tree_flatten(out_batched)
  assert out_tree1 == out_tree, "todo error"
  assert out_tree2 == out_tree, "todo error"
  yield out, [0 if b else not_mapped for b in batched]


def custom_primitive_vmap(name: str, args, dims, call_jaxpr: core.ClosedJaxpr,
                          jvp: Rule | None, vjp_fwd: Rule | None,
                          vjp_bwd: Rule | None, res_tree, transpose: Rule | None,
                          vmap: Rule | None, **_):
  del call_jaxpr, jvp, vjp_fwd, vjp_bwd, transpose
  if vmap is None:
    raise NotImplementedError(
        f"'vmap' not implemented for custom primitive '{name}'")
  return vmap.call_wrapped(args, dims)


# def vectorized_vmap(fun, batched, *args, **kwargs):
#   batched_flat, _ = tree_flatten(batched)
#   assert all(batched_flat[0] == b for b in batched_flat[1:]), "todo err"
#   out = fun(*args, **kwargs)
#   return out, tree_map(lambda _: batched_flat[0], tree_structure(out))
