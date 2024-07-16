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

import inspect
from collections.abc import Sequence, Callable
from functools import partial, update_wrapper
from typing import Any, Generic, TypeVar

from jax._src import ad_util
from jax._src import api_util
from jax._src import core
from jax._src import custom_api_util
from jax._src import linear_util as lu
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src.interpreters import ad
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.tree_util import (
  tree_flatten,
  tree_unflatten,
  treedef_children,
  tree_map,
  tree_flatten_with_path,
  keystr,
)
from jax._src.util import safe_map, safe_zip, split_list, unzip2

map = safe_map
zip = safe_zip

T = TypeVar("T")


class custom_transformations(Generic[T]):
  fun: Callable[..., T]
  nondiff_argnums: set[int]
  jvp: Callable[..., tuple[T, T]] | None = None
  vjp_fwd: Callable[..., tuple[T, Any]] | None = None
  vjp_bwd: Callable[..., tuple[Any, ...]] | None = None
  lin: Callable[..., T] | None = None
  # vmap: Callable[..., tuple[T, bool | Sequence[bool]]]

  def __init__(
    self,
    fun: Callable[..., T],
    nondiff_argnums: Sequence[int] = (),
  ):
    update_wrapper(self, fun)
    self.fun = fun
    self.nondiff_argnums = set(nondiff_argnums)

  __getattr__ = custom_api_util.forward_attr

  def def_jvp(self, jvp: Callable[..., tuple[T, T]]) -> None:
    self.jvp = jvp

  def def_vjp(
    self, fwd: Callable[..., tuple[T, Any]], bwd: Callable[..., tuple[Any, ...]]
  ) -> None:
    self.vjp_fwd = fwd
    self.vjp_bwd = bwd

  def def_lin(self, lin: Callable[..., T]) -> None:
    self.lin = lin

  # def def_vmap(self, vmap: Callable[..., tuple[T, bool | Sequence[bool]]]) -> None:
  #   self.vmap = vmap

  @traceback_util.api_boundary
  def __call__(self, *args: Any, **kwargs: Any) -> T:
    assert not (self.nondiff_argnums), "todo"

    name = getattr(self.fun, "__name__", str(self.fun))

    # trace the function to a jaxpr
    args = _resolve_kwargs(self.fun, args, kwargs)
    args_flat, tree_in = tree_flatten(args)
    fun_flat, tree_out_thunk = api_util.flatten_fun_nokwargs(
      lu.wrap_init(self.fun), tree_in
    )
    avals_in = [core.raise_to_shaped(core.get_aval(x)) for x in args_flat]
    debug = pe.debug_info(
      self.fun, tree_in, tree_out_thunk, False, "custom_transformations"
    )
    jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(fun_flat, avals_in, debug)
    fun_jaxpr = core.ClosedJaxpr(pe.convert_constvars_jaxpr(jaxpr), ())
    tree_out = tree_out_thunk()

    jvp = self.jvp
    if jvp:
      rule_name = getattr(jvp, "__name__", str(jvp))
      jvp = _flatten_jvp(
        lu.wrap_init(jvp), name, rule_name, tree_in, tree_out, fun_jaxpr.out_avals
      )

    vjp_fwd = self.vjp_fwd
    if vjp_fwd:
      vjp_bwd = self.vjp_bwd
      if not vjp_bwd:
        raise ValueError("TODO")
      fwd_name = getattr(vjp_fwd, "__name__", str(vjp_fwd))
      vjp_fwd, tree_res_thunk = _flatten_vjp_fwd(
        lu.wrap_init(vjp_fwd), name, fwd_name, tree_in, tree_out
      )
      bwd_name = getattr(vjp_bwd, "__name__", str(vjp_bwd))
      vjp_bwd = _flatten_vjp_bwd(
        lu.wrap_init(vjp_bwd),
        name,
        bwd_name,
        avals_in,
        tree_in,
        tree_out,
        tree_res_thunk,
      )

    lin = self.lin
    if lin:
      rule_name = getattr(lin, "__name__", str(lin))
      lin = _flatten_lin(
        lu.wrap_init(lin), name, rule_name, tree_in, tree_out, tree_res_thunk
      )

    out_flat = custom_transformations_p.bind(
      *consts,
      *args_flat,
      num_consts=len(consts),
      name=name,
      prim_jaxpr=fun_jaxpr,
      jvp=jvp,
      fwd=vjp_fwd,
      bwd=vjp_bwd,
      lin=lin,
    )
    return tree_unflatten(tree_out, out_flat)


def _resolve_kwargs(fun, args, kwargs):
  if isinstance(fun, partial):
    fun = lambda *_, **__: None
  ba = inspect.signature(fun).bind(*args, **kwargs)
  ba.apply_defaults()
  if ba.kwargs:
    raise TypeError("keyword arguments could not be resolved to positions")
  else:
    return ba.args


@lu.transformation
def _flatten_jvp(name, rule_name, tree_in, tree_out, avals_out, *args):
  primals_in, tangents_in = split_list(args, [len(args) // 2])
  py_primals = tree_unflatten(tree_in, primals_in)
  py_tangents = tree_unflatten(tree_in, tangents_in)
  pair_out = yield (py_primals, py_tangents), {}
  if not isinstance(pair_out, (list, tuple)) or len(pair_out) != 2:
    msg = (
      f"Custom JVP rule {rule_name} for function {name} "
      "must produce a pair (list or tuple of length two) representing "
      f"primal and tangent outputs, but got {pair_out}."
    )
    raise TypeError(msg)
  py_primals_out, py_tangents_out = pair_out
  primals_out, tree_prim = tree_flatten(py_primals_out)
  tangents_out, tree_tan = tree_flatten(py_tangents_out)
  if tree_prim != tree_tan:
    msg = (
      f"Custom JVP rule {rule_name} for function {name} must "
      "produce primal and tangent outputs with equal container (pytree) "
      f"structures, but got {tree_prim} and {tree_tan} respectively."
    )
    raise TypeError(msg)
  # TODO(dfm): Compare primal and tanget avals
  if tree_prim != tree_out:
    msg = (
      f"Custom JVP rule {rule_name} for function {name} must "
      "produce primal and tangent outputs with the same container "
      "(pytree) structure as the primal function output, but got "
      f"{tree_prim} and {tree_out} respectively."
    )
    raise TypeError(msg)
  avals_prim = [core.raise_to_shaped(core.get_aval(x)) for x in primals_out]
  if not all(map(core.typematch, avals_prim, avals_out)):
    msg = "TODO"
    raise TypeError(msg)
  yield (*primals_out, *tangents_out)


@lu.transformation_with_aux
def _flatten_vjp_fwd(name, rule_name, tree_in, tree_out, *args):
  py_args = tree_unflatten(tree_in, args)
  pair_out = yield py_args, {}
  if not isinstance(pair_out, (list, tuple)) or len(pair_out) != 2:
    msg = (
      f"Custom VJP fwd rule {rule_name} for function {name} "
      "must produce a pair (list or tuple of length two) where the first "
      "element represents the primal output (equal to those of the "
      f"custom_vjp-decorated function {name}) and the "
      "second element represents residuals (i.e. values stored from the "
      "forward pass for use on the backward pass), but "
      f"instead of a pair the fwd rule {rule_name} produced {pair_out}."
    )
    raise TypeError(msg)
  py_primals_out, py_res = pair_out
  primals_out, tree_prim = tree_flatten(py_primals_out)
  assert tree_prim == tree_out, "TODO"
  # TODO(dfm): Check trees and avals, raising better errors
  res, tree_res = tree_flatten(py_res)
  yield (*res, *primals_out), tree_res


@lu.transformation
def _flatten_vjp_bwd(
  name, rule_name, avals_in, tree_in, tree_out, tree_res_thunk, *args
):
  tree_res = tree_res_thunk()
  assert len(args) == tree_res.num_leaves + tree_out.num_leaves
  res, cts_out = split_list(args, [tree_res.num_leaves])
  py_res = tree_unflatten(tree_res, res)
  py_cts_out = tree_unflatten(tree_out, cts_out)
  py_cts_in = yield (py_res, py_cts_out), {}
  if isinstance(py_cts_in, list) and len(py_cts_in) == len(treedef_children(tree_in)):
    py_cts_in = tuple(py_cts_in)
  # For each None in py_cts_in, indicating an argument for which the rule
  # produces no cotangent, we replace it with a pytree with the structure of the
  # corresponding subtree of in_tree and with leaves of a non-pytree sentinel
  # object, to be replaced with Nones in the final returned result.
  zero = object()  # non-pytree sentinel to replace Nones in py_cts_in
  dummy = tree_unflatten(tree_in, [object()] * tree_in.num_leaves)
  keypaths, _ = unzip2(tree_flatten_with_path(dummy)[0])
  cts_in_flat = []

  def append(x, d):
    num_leaves = len(tree_flatten(d)[0])
    if x is None and d is not None:
      cts_in_flat.extend([zero] * num_leaves)
    elif x is not None:
      cts_in_flat.extend([x] * num_leaves)
    return x

  try:
    if not isinstance(py_cts_in, tuple):
      raise ValueError
    tree_map(append, py_cts_in, dummy, is_leaf=lambda x: x is None)
  except ValueError:
    _, tree_in2 = tree_flatten(py_cts_in)
    msg = (
      f"Custom VJP bwd rule {rule_name} for function {name} must produce "
      "an output with the same container (pytree) structure as the args "
      "tuple of the primal function, and in particular must produce a "
      "tuple of length equal to the number of arguments to the primal "
      f"function, but got bwd output structure {tree_in2} for primal "
      f"input structure {tree_in}."
    )
    raise TypeError(msg) from None
  results = []
  for kp, a, ct in zip(keypaths, avals_in, cts_in_flat):
    if ct is zero or a != a.at_least_vspace():
      results.append(ad_util.Zero(a.at_least_vspace()))
    elif type(ct) is ad_util.SymbolicZero:
      # if not core.typecompat(a.at_least_vspace(), a_ := ct.aval):
      #   msg = (
      #     "Custom VJP bwd rule produced a SymbolicZero with a shape/dtype "
      #     "that does not match the corresponding input tangent shape/dtype: "
      #     f"at output{keystr(kp)} the SymbolicZero had shape/dtype "
      #     f"{a_.str_short()} while the "
      #     f"corresponding input had shape/dtype {a.str_short()}. "
      #     "Consider just returning a None here instead of a SymbolicZero "
      #     "object."
      #   )
      #   raise ValueError(msg)
      results.append(ad_util.Zero(ct.aval))
    else:
      # if not core.typecompat(a.at_least_vspace(), a_ := core.get_aval(ct)) and not (
      #   _temporary_dtype_exception(a, a_) or _temporary_shape_exception(a, a_)
      # ):
      #   msg = (
      #     "Custom VJP bwd rule must produce an output with the same "
      #     "shape/dtypes as the args tuple of the primal function, but at "
      #     f"output{keystr(kp)} the bwd rule produced an output of "
      #     f"shape/dtype {raise_to_shaped(a_).str_short()} corresponding "
      #     f"to an input of shape/dtype {a.str_short()}."
      #   )
      #   raise ValueError(msg)
      results.append(ct)
  yield results


@lu.transformation
def _flatten_lin(name, rule_name, tree_in, tree_out, tree_res_thunk, *args):
  tree_res = tree_res_thunk()
  res, tangents_in = split_list(args, [tree_res.num_leaves])
  py_res = tree_unflatten(tree_res, res)
  py_tangents_in = tree_unflatten(tree_in, tangents_in)
  py_tangents_out = yield (py_res, *py_tangents_in), {}
  tangents_out, tree_tan = tree_flatten(py_tangents_out)
  if tree_tan != tree_out:
    msg = (
      f"Custom lin rule {rule_name} for function {name} must "
      "produce tangent outputs with the same container "
      "(pytree) structure as the primal function output, but got "
      f"{tree_tan} and {tree_out} respectively."
    )
    raise TypeError(msg)
  yield tangents_out


def _custom_transformations_impl(*args, prim_jaxpr: core.ClosedJaxpr, **_):
  return core.jaxpr_as_fun(prim_jaxpr)(*args)


def _custom_transformations_abstract_eval(*args, prim_jaxpr: core.ClosedJaxpr, **_):
  del args  # unused
  return prim_jaxpr.out_avals


def _custom_transformations_jvp(
  primals,
  tangents,
  *,
  num_consts: int,
  name: str,
  prim_jaxpr: core.ClosedJaxpr,
  jvp: lu.WrappedFun | None,
  fwd: lu.WrappedFun | None,
  bwd: lu.WrappedFun | None,
  lin: lu.WrappedFun | None,
):
  if jvp is None:
    raise NotImplementedError(f"JVP rule is not defined for function {name}")
  assert all(isinstance(t, ad.Zero) for t in tangents[:num_consts])
  consts, primals_in = split_list(primals, [num_consts])
  _, tangents_in = split_list(tangents, [num_consts])
  tangents_in = map(ad.instantiate_zeros, tangents_in)
  out_flat = jvp_helper_p.bind(
    *consts,
    *primals_in,
    *tangents_in,
    num_consts=num_consts,
    name=name,
    prim_jaxpr=prim_jaxpr,
    jvp=jvp,
    fwd=fwd,
    bwd=bwd,
    lin=lin,
  )
  primals_out, tangents_out = split_list(out_flat, [len(out_flat) // 2])
  return primals_out, tangents_out


custom_transformations_p = core.Primitive("custom_transformations")
custom_transformations_p.multiple_results = True
custom_transformations_p.def_impl(_custom_transformations_impl)
custom_transformations_p.def_abstract_eval(_custom_transformations_abstract_eval)
ad.primitive_jvps[custom_transformations_p] = _custom_transformations_jvp
mlir.register_lowering(
  custom_transformations_p,
  mlir.lower_fun(_custom_transformations_impl, multiple_results=True),
)


def _jvp_helper_impl(*args, num_consts: int, jvp, **_):
  _, args = split_list(args, [num_consts])
  return jvp.call_wrapped(*args)


def _jvp_helper_abstract_eval(*args, prim_jaxpr: core.ClosedJaxpr, **_):
  del args  # unused
  out_avals = prim_jaxpr.out_avals
  return (*out_avals, *out_avals)


def _jvp_helper_partial_eval(
  trace,
  *tracers,
  num_consts: int,
  name: str,
  prim_jaxpr: core.ClosedJaxpr,
  jvp: lu.WrappedFun | None,
  fwd: lu.WrappedFun | None,
  bwd: lu.WrappedFun | None,
  lin: lu.WrappedFun | None,
):
  consts, tracers = split_list(tracers, [num_consts])
  assert all(t.pval.is_known() for t in consts)
  primals, tangents = split_list(tracers, [len(tracers) // 2])
  assert all(t.pval.is_known() for t in primals)
  tangents = [trace.instantiate_const(t) if t.pval.is_known() else t for t in tangents]
  assert all(not t.pval.is_known() for t in tangents)
  out_avals = prim_jaxpr.out_avals

  knowns = [t.pval.get_known() for t in primals]
  primals_out_and_res = fwd.call_wrapped(*knowns)
  res, primals_out = split_list(
    primals_out_and_res, [len(primals_out_and_res) - len(out_avals)]
  )

  res = map(trace.new_instantiated_const, res)
  consts = map(trace.instantiate_const, consts)
  primals = map(trace.instantiate_const, primals)
  tangents_out = [
    pe.JaxprTracer(trace, pe.PartialVal.unknown(a), None) for a in out_avals
  ]
  name_stack = source_info_util.current_name_stack()[len(trace.name_stack) :]
  source = source_info_util.current().replace(name_stack=name_stack)
  eqn = pe.new_eqn_recipe(
    [*res, *consts, *primals, *tangents],
    tangents_out,
    transpose_helper_p,
    dict(
      num_res=len(res),
      num_consts=num_consts,
      name=name,
      prim_jaxpr=prim_jaxpr,
      jvp=jvp,
      fwd=fwd,
      bwd=bwd,
      lin=lin,
    ),
    prim_jaxpr.effects,
    source,
  )
  for t in tangents_out:
    t.recipe = eqn

  return primals_out + tangents_out


def _jvp_helper_dce(used_outputs: list[bool], eqn: core.JaxprEqn):
  primals_out_used, tangents_out_used = split_list(
    used_outputs, [len(used_outputs) // 2]
  )
  if any(tangents_out_used):
    return [True] * len(eqn.invars), eqn
  prim_jaxpr = eqn.params["prim_jaxpr"]
  prim_jaxpr, primals_in_used = pe.dce_jaxpr(prim_jaxpr.jaxpr, primals_out_used)
  prim_jaxpr = core.ClosedJaxpr(prim_jaxpr, ())

  invars = eqn.invars[: len(primals_in_used)]
  invars = [v for v, used in zip(invars, primals_in_used) if used]
  outvars = eqn.outvars[: len(primals_out_used)]
  outvars = [v for v, used in zip(outvars, primals_out_used) if used]
  new_eqn = pe.new_jaxpr_eqn(
    invars,
    outvars,
    custom_transformations_p,
    dict(
      num_consts=eqn.params["num_consts"],
      name=eqn.params["name"],
      prim_jaxpr=prim_jaxpr,
      jvp=eqn.params["jvp"],
      fwd=eqn.params["fwd"],
      bwd=eqn.params["bwd"],
      lin=eqn.params["lin"],
    ),
    prim_jaxpr.effects,
    eqn.source_info,
  )
  num_tangents = len(primals_in_used) - eqn.params["num_consts"]
  return primals_in_used + [False] * num_tangents, new_eqn


jvp_helper_p = core.Primitive("jvp_helper")
jvp_helper_p.multiple_results = True
jvp_helper_p.def_impl(_jvp_helper_impl)
jvp_helper_p.def_abstract_eval(_jvp_helper_abstract_eval)
mlir.register_lowering(
  jvp_helper_p,
  mlir.lower_fun(_jvp_helper_impl, multiple_results=True),
)
pe.custom_partial_eval_rules[jvp_helper_p] = _jvp_helper_partial_eval
pe.dce_rules[jvp_helper_p] = _jvp_helper_dce


def _transpose_helper_impl(
  *args,
  num_res: int,
  num_consts: int,
  name: str,
  prim_jaxpr: core.ClosedJaxpr,
  jvp: lu.WrappedFun | None,
  fwd: lu.WrappedFun | None,
  bwd: lu.WrappedFun | None,
  lin: lu.WrappedFun | None,
):
  del name, prim_jaxpr, fwd, bwd
  if lin is None:
    # If a custom lin rule isn't provided, use the JVP rule. This is inefficient,
    # but it's the best we can do by default.
    _, args = split_list(args, [num_res + num_consts])
    primals, tangents = split_list(args, [len(args) // 2])
    primals_and_tangents_out = jvp.call_wrapped(*primals, *tangents)
    _, tangents_out = split_list(
      primals_and_tangents_out, [len(primals_and_tangents_out) // 2]
    )
    return tangents_out
  else:
    res, _, args = split_list(args, [num_res, num_consts])
    _, tangents_in = split_list(args, [len(args) // 2])
    tangents_out = lin.call_wrapped(*res, *tangents_in)
    return tangents_out


def _transpose_helper_abstract_eval(*_, prim_jaxpr: core.ClosedJaxpr, **__):
  return prim_jaxpr.out_avals


def _transpose_helper_transpose(
  cts_out,
  *args,
  num_res: int,
  num_consts: int,
  name: str,
  prim_jaxpr: core.ClosedJaxpr,
  jvp: lu.WrappedFun | None,
  fwd: lu.WrappedFun | None,
  bwd: lu.WrappedFun | None,
  lin: lu.WrappedFun | None,
):
  del name, num_consts, jvp, fwd, lin
  res, _ = split_list(args, [num_res])
  cts_in = bwd.call_wrapped(*res, *cts_out)
  return [None] * (num_res + len(prim_jaxpr.in_avals)) + cts_in


transpose_helper_p = core.Primitive("transpose_helper")
transpose_helper_p.multiple_results = True
transpose_helper_p.def_impl(_transpose_helper_impl)
transpose_helper_p.def_abstract_eval(_transpose_helper_abstract_eval)
mlir.register_lowering(
  transpose_helper_p,
  mlir.lower_fun(_transpose_helper_impl, multiple_results=True),
)
ad.primitive_transposes[transpose_helper_p] = _transpose_helper_transpose
