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
from weakref import ref

from jax._src import ad_util
from jax._src import api_util
from jax._src import core
from jax._src import custom_api_util
from jax._src import linear_util as lu
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import custom_derivatives as cd
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.tree_util import (tree_flatten, tree_unflatten, treedef_children,
                                tree_map, tree_flatten_with_path, keystr)
from jax._src.util import safe_map, safe_zip, split_list, unzip2

map = safe_map
zip = safe_zip

T = TypeVar("T")


class custom_ad(Generic[T]):
  fun: Callable[..., T]
  nondiff_argnums: Sequence[int]
  jvp: Callable[..., tuple[T, T]] | None = None
  fwd: Callable[..., tuple[T, Any]] | None = None
  bwd: Callable[..., tuple[Any, ...]] | None = None
  lin: Callable[..., T] | None = None

  def __init__(
      self,
      fun: Callable[..., T],
      nondiff_argnums: Sequence[int] = (),
  ):
    update_wrapper(self, fun)
    self.fun = fun
    self.nondiff_argnums = nondiff_argnums

  __getattr__ = custom_api_util.forward_attr

  def defjvp(self, jvp: Callable[..., tuple[T, T]]) -> None:
    self.jvp = jvp
    return jvp

  def defjvps(self, *jvps: Callable[..., T] | None):
    if self.nondiff_argnums:
      raise TypeError("Can't use ``defjvps`` with ``nondiff_argnums``.")

    def jvp(primals, tangents):
      primal_out = self(*primals)
      zeros = cd._zeros_like_pytree(primal_out)
      all_tangents_out = [jvp(t, primal_out, *primals) if jvp else zeros
                          for t, jvp in zip(tangents, jvps)]
      tangent_out = tree_map(cd._sum_tangents, primal_out, *all_tangents_out)
      return primal_out, tangent_out

    self.defjvp(jvp)

  def deffwd(self, fwd: Callable[..., tuple[T, Any]]) -> None:
    self.fwd = fwd
    return fwd

  def defbwd(self, bwd: Callable[..., tuple[Any, ...]]) -> None:
    self.bwd = bwd
    return bwd

  def deflin(self, lin: Callable[..., T]) -> None:
    self.lin = lin
    return lin

  @traceback_util.api_boundary
  def __call__(self, *args: Any, **kwargs: Any) -> T:
    if all(f is None for f in (self.jvp, self.fwd, self.bwd, self.lin)):
      return self.fun(*args, **kwargs)

    name = getattr(self.fun, "__name__", str(self.fun))
    args = _resolve_kwargs(self.fun, args, kwargs)

    if self.nondiff_argnums:
      for i in self.nondiff_argnums: cd._check_for_tracers(args[i])
      nondiff_argnums = set(self.nondiff_argnums)
      dyn_argnums = [i for i in range(len(args)) if i not in nondiff_argnums]
      fun, dyn_args = api_util.argnums_partial(
          lu.wrap_init(self.fun), dyn_argnums, args,
          require_static_args_hashable=False)
      # Note: Here we sort nondiff_argnums, but in custom_jvp and custom_vjp we
      # don't.
      static_args = [args[i] for i in sorted(self.nondiff_argnums)]
    else:
      fun, dyn_args = lu.wrap_init(self.fun), args
      static_args = []

    # Since this is an "initial style" primitive, we trace the primal function
    # to a jaxpr right off the bat. This should be updated eventually, but for
    # now it significantly simplifies the implementation.
    args_flat, tree_in = tree_flatten(args)
    fun_flat, tree_out_thunk = api_util.flatten_fun_nokwargs(fun, tree_in)
    avals_in = [core.raise_to_shaped(core.get_aval(x)) for x in args_flat]
    debug = pe.debug_info(fun, tree_in, tree_out_thunk, False, "custom_ad")
    jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(fun_flat, avals_in, debug)
    prim_jaxpr = core.ClosedJaxpr(pe.convert_constvars_jaxpr(jaxpr), ())
    tree_out = tree_out_thunk()

    if self.jvp:
      jvp = self.jvp
      rule_name = getattr(jvp, "__name__", str(jvp))
      jvp = _flatten_jvp(
          lu.wrap_init(jvp), name, rule_name, tree_in, tree_out,
          prim_jaxpr.out_avals)
    else:
      # Default to JVP of primal (ok because we already traced it)
      @lu.wrap_init
      def jvp(*args):
        _, tangents = split_list(args, [len(args) // 2])
        nz = [False] * len(consts)
        nz += [not isinstance(t, ad_util.Zero) for t in tangents]
        del tangents
        jvp_jaxpr, _ = ad.jvp_jaxpr(prim_jaxpr, nz, False)
        return core.jaxpr_as_fun(jvp_jaxpr)(*consts, *args)

    fwd = self.fwd
    if fwd:
      fwd_name = getattr(fwd, "__name__", str(fwd))
      fwd, res_type_thunk = _flatten_fwd(
          lu.wrap_init(fwd), name, fwd_name, tree_in, tree_out,
          prim_jaxpr.out_avals)
    else:
      res_type_thunk = None

    bwd = self.bwd
    if bwd:
      if fwd is None:
        raise AttributeError(
            f"No `fwd` rule defined for the custom_ad function {name}. When a "
            "`bwd` rule is defined, a `fwd` rule must also be set using "
            "`def_fwd`.")
      bwd_name = getattr(bwd, "__name__", str(bwd))
      bwd = _flatten_bwd(lu.wrap_init(bwd), name, bwd_name, avals_in, tree_in,
          tree_out, res_type_thunk)

    lin = self.lin
    if lin:
      if fwd is None:
        raise AttributeError(
            f"No `fwd` rule defined for the custom_ad function {name}. When a "
            "`lin` rule is defined, a `fwd` rule must also be set using "
            "`def_fwd`.")
      rule_name = getattr(lin, "__name__", str(lin))
      lin = _flatten_lin(
          lu.wrap_init(lin), name, rule_name, tree_in, tree_out,
          prim_jaxpr.out_avals, res_type_thunk)

    if lin or bwd:
      if not bwd:
        # Default to transposing lin
        lin_in_avals = prim_jaxpr.in_avals[len(consts):]
        bwd = _transpose_lin_or_bwd(lin, res_type_thunk, lin_in_avals)

      if not lin:
        # TODO(dfm): transpose bwd to get a default lin.
        # assert False, "TODO"
        pass

    else:
      if fwd:
        raise ValueError(
            f"A `fwd` rule was defined for the custom_ad function {name}, but "
            "since no `lin` or `bwd` rule was defined, the `fwd` will not be "
            "used.")

    jvp = cd._add_args(jvp, static_args)
    fwd = cd._add_args(fwd, static_args) if fwd else None
    bwd = cd._add_args(bwd, static_args) if bwd else None
    lin = cd._add_args(lin, static_args) if lin else None

    # Because this is an "initial style" primitive, we also want to stage out
    # all the rules (except for bwd), but we do that lazily to avoid infinite
    # recursions and to avoid evaluating the rules until we need them.
    # @pe._memoize
    def jvp_jaxpr_thunk():
      jvp_avals_in = (*avals_in, *avals_in)
      jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(jvp, jvp_avals_in)
      return jaxpr, consts

    # @pe._memoize
    def fwd_jaxpr_thunk():
      jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(fwd, avals_in)
      return jaxpr, consts

    # TODO(dfm): Think about whether or not we need to stage out bwd and lin.
    # @pe._memoize
    # def lin_jaxpr_thunk():
    #   _, avals_res = res_type_thunk()
    #   lin_in_avals = (*avals_res, *avals_in)
    #   jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(lin, lin_in_avals)
    #   return jaxpr, consts

    # @pe._memoize
    # def bwd_jaxpr_thunk():
    #   _, avals_res = res_type_thunk()
    #   bwd_in_avals = (*avals_res, *prim_jaxpr.in_avals)
    #   jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(bwd, bwd_in_avals)
    #   return jaxpr, consts

    out_flat = custom_ad_p.bind(*consts, *args_flat, num_consts=len(consts),
        name=name, prim_jaxpr=prim_jaxpr, jvp_jaxpr_thunk=jvp_jaxpr_thunk,
        fwd_jaxpr_thunk=fwd_jaxpr_thunk, bwd=bwd, lin=lin)
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
  avals_prim = [core.raise_to_shaped(core.get_aval(x)) for x in primals_out]
  jvp_ty_tree = tree_unflatten(tree_out, [a.str_short() for a in avals_prim])
  prim_ty_tree = tree_unflatten(tree_out, [a.str_short() for a in avals_out])
  if tree_prim != tree_out:
    msg = (f"Custom JVP rule {rule_name} for function {name} must produce a "
           "pair (list or tuple of length two) where the first element "
           "represents the primal output (equal in value to the output of the "
           f"function {name}, and in particular of the same pytree structure), "
           "but instead the rule output's first element had pytree structure:\n"
           f"""    {str(jvp_ty_tree).replace("'", "")}\n"""
           f"while the function {name} had output pytree structure:\n"
           f"""    {str(prim_ty_tree).replace("'", "")}.""")
    raise TypeError(msg)
  if not all(map(core.typematch, avals_prim, avals_out)):
    msg = (f"Custom JVP rule {rule_name} for function {name} must "
           "produce a pair (list or tuple of length two) "
           "where the first element represents the primal output "
           f"(equal in value to the output of the function {name}, and in "
           "particular with leaves of the same shape/dtype), but instead the "
           "rule output's first element had shapes/dtypes of:\n"
           f"""    {str(jvp_ty_tree).replace("'", "")}\n"""
           f"while the function {name} had output shapes/dtypes of:\n"
           f"""    {str(prim_ty_tree).replace("'", "")}""")
    raise TypeError(msg)
  primal_avals_out = [core.raise_to_shaped(core.get_aval(x), weak_type=False)
                      for x in primals_out]
  tangent_avals_out = [core.raise_to_shaped(core.get_aval(t), weak_type=False)
                       # TODO(dfm): Add this back in when supporting symbolic zeros
                       # if type(t) is not SymbolicZero else t.aval.strip_weak_type()
                       for t in tangents_out]
  if primal_avals_out != tangent_avals_out:
    disagreements = "\n".join(
        f"  primal {av1.str_short()} for tangent {av2.str_short()}"
        for av1, av2 in zip(primal_avals_out, tangent_avals_out) if av1 != av2)
    msg = ("Custom JVP rule must produce primal and tangent outputs with equal "
           f"shapes and dtypes, but {rule_name} returned:\n{disagreements}")
    raise TypeError(msg)
  yield (*primals_out, *tangents_out)


@lu.transformation_with_aux
def _flatten_fwd(name, rule_name, tree_in, tree_out, avals_out, *args):
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
  avals_prim = [core.raise_to_shaped(core.get_aval(x)) for x in primals_out]
  fwd_ty_tree = tree_unflatten(tree_out, [a.str_short() for a in avals_prim])
  prim_ty_tree = tree_unflatten(tree_out, [a.str_short() for a in avals_out])
  if tree_prim != tree_out:
    msg = (f"Custom VJP fwd rule {rule_name} for function {name} must produce "
           "a pair (list or tuple of length two) where the first element "
           "represents the primal output (equal in value to the output of the "
           f"function {name}, and in particular of the same pytree structure), "
           "but instead the rule output's first element had pytree structure:\n"
           f"""    {str(fwd_ty_tree).replace("'", "")}\n"""
           f"while the function {name} had output pytree structure:\n"
           f"""    {str(prim_ty_tree).replace("'", "")}.""")
    raise TypeError(msg)
  if not all(map(core.typematch, avals_prim, avals_out)):
    msg = (f"Custom VJP fwd rule {rule_name} for function {name} must "
           "produce a pair (list or tuple of length two) "
           "where the first element represents the primal output "
           f"(equal in value to the output of the function {name}, "
           "and in particular with leaves of the same shape/dtype), but "
           "instead the rule output's first element had shapes/dtypes of:\n"
           f"""    {str(fwd_ty_tree).replace("'", "")}\n"""
           f"while the function {name} had output shapes/dtypes of:\n"
           f"""    {str(prim_ty_tree).replace("'", "")}""")
    raise TypeError(msg)
  res, tree_res = tree_flatten(py_res)
  avals_res = [core.raise_to_shaped(core.get_aval(x)) for x in res]
  yield (*res, *primals_out), (tree_res, avals_res)


@lu.transformation
def _flatten_bwd(name, rule_name, avals_in, tree_in, tree_out, res_type_thunk, *args):
  tree_res, _ = res_type_thunk()
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
      "an output with the same pytree structure as the args tuple of the "
      f"primal function {name}, and in particular must produce a tuple of "
      "length equal to the number of arguments to the primal "
      f"function, but got bwd output structure {tree_in2} for primal "
      f"input structure {tree_in}."
    )
    raise TypeError(msg) from None
  results = []
  for kp, a, ct in zip(keypaths, avals_in, cts_in_flat):
    if ct is zero or a != a.at_least_vspace():
      results.append(ad_util.Zero(a.at_least_vspace()))
    elif type(ct) is ad_util.SymbolicZero:
      # TODO(dfm): Add this back in when supporting symbolic zeros.
      assert 0, "TODO"
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
      # results.append(ad_util.Zero(ct.aval))
    else:
      if (not core.typecompat(a.at_least_vspace(), a_ := core.get_aval(ct))
          and not (cd._temporary_dtype_exception(a, a_) or
                   cd._temporary_shape_exception(a, a_))):
        msg = (
          "Custom VJP bwd rule must produce an output with the same "
          f"shape/dtypes as the args tuple of the primal function {name}, but "
          f"at output{keystr(kp)} the bwd rule produced an output of "
          f"shape/dtype {core.raise_to_shaped(a_).str_short()} corresponding "
          f"to an input of shape/dtype {a.str_short()}.")
        raise ValueError(msg)
      results.append(ct)
  yield results


@lu.transformation
def _flatten_lin(name, rule_name, tree_in, tree_out, avals_out, res_type_thunk,
                 *args):
  tree_res, _ = res_type_thunk()
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
      f"{tree_tan} and {tree_out} respectively.")
    raise TypeError(msg)
  tangent_avals_out = [core.raise_to_shaped(core.get_aval(t), weak_type=False)
                       # TODO(dfm): Add this back in when supporting symbolic zeros
                       # if type(t) is not SymbolicZero else t.aval.strip_weak_type()
                       for t in tangents_out]
  if avals_out != tangent_avals_out:
    disagreements = "\n".join(
        f"  primal {av1.str_short()} for tangent {av2.str_short()}"
        for av1, av2 in zip(avals_out, tangent_avals_out) if av1 != av2)
    msg = ("Custom lin rule must produce primal and tangent outputs with equal "
           f"shapes and dtypes, but {rule_name} returned:\n{disagreements}")
    raise TypeError(msg)

  yield tangents_out


def _transpose_lin_or_bwd(fun: lu.WrappedFun, res_type_thunk, in_avals):
  @lu.wrap_init
  def transposed_fun(*args):
    tree_res, _ = res_type_thunk()
    res, cts_out = split_list(args, [tree_res.num_leaves])

    in_avals_ = [core.raise_to_shaped(core.get_aval(x)) for x in res] + in_avals
    jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(fun, in_avals_)
    # TODO(dfm): Do DCE here like in linear_transpose?

    dummies = res + [ad.UndefinedPrimal(x) for x in in_avals]
    cts_in = ad.backward_pass(jaxpr, True, consts, dummies, cts_out)
    cts_res, cts_in = split_list(cts_in, [len(res)])
    assert all(isinstance(ct, ad_util.Zero) for ct in cts_res)
    cts_in = map(ad.instantiate_zeros, cts_in)
    return cts_in

  return transposed_fun

def _custom_ad_impl(*args, prim_jaxpr: core.ClosedJaxpr, **_):
  return core.jaxpr_as_fun(prim_jaxpr)(*args)

def _custom_ad_abstract_eval(*args, prim_jaxpr: core.ClosedJaxpr, **_):
  del args  # unused
  return prim_jaxpr.out_avals

def _custom_ad_jvp(primals, tangents, *, num_consts: int, name: str,
    prim_jaxpr: core.ClosedJaxpr, jvp_jaxpr_thunk: Callable,
    fwd_jaxpr_thunk: Callable | None, bwd: lu.WrappedFun | None,
    lin: lu.WrappedFun | None):
  assert all(isinstance(t, ad.Zero) for t in tangents[:num_consts])
  consts, primals_in = split_list(primals, [num_consts])
  _, tangents_in = split_list(tangents, [num_consts])
  tangents_in = map(ad.instantiate_zeros, tangents_in)
  if bwd is None and lin is None:
    jvp_jaxpr, jvp_consts = jvp_jaxpr_thunk()
    closed_jvp_jaxpr = pe.close_jaxpr(pe.convert_constvars_jaxpr(jvp_jaxpr))
    out_flat = core.call(lu.wrap_init(core.jaxpr_as_fun(closed_jvp_jaxpr)),
                         *jvp_consts, *primals_in, *tangents_in)
  else:
    out_flat = jvp_helper_p.bind(*consts, *primals_in, *tangents_in,
                                 num_consts=num_consts, name=name,
                                 prim_jaxpr=prim_jaxpr,
                                 jvp_jaxpr_thunk=jvp_jaxpr_thunk,
                                 fwd_jaxpr_thunk=fwd_jaxpr_thunk,
                                 bwd=bwd, lin=lin)
  primals_out, tangents_out = split_list(out_flat, [len(out_flat) // 2])
  return primals_out, tangents_out

def _custom_ad_transpose(ct, *args, prim_jaxpr: core.ClosedJaxpr, **_):
  return ad.backward_pass(prim_jaxpr.jaxpr, None, prim_jaxpr.consts, args, ct)


def _custom_ad_batching(spmd_axis_name, axis_size, axis_name, main_type, args,
                        dims, num_consts: int, name: str,
                        prim_jaxpr: core.ClosedJaxpr, jvp_jaxpr_thunk: Callable,
                        fwd_jaxpr_thunk: Callable | None,
                        bwd: lu.WrappedFun | None,
                        lin: lu.WrappedFun | None):
  args = [batching.moveaxis(x, d, 0) if d is not batching.not_mapped and d != 0
          else x for x, d in zip(args, dims)]
  in_batched = [d is not batching.not_mapped for d in dims]
  prim_jaxpr_batched, prim_out_batched = batching.batch_jaxpr(
      prim_jaxpr, axis_size, in_batched, False, axis_name, spmd_axis_name,
      main_type)
  out_dims1 = [0 if b else batching.not_mapped for b in prim_out_batched]
  out_dims2 = []

  _, primals_in_batched = split_list(in_batched, [num_consts])

  # @pe._memoize
  def jvp_jaxpr_thunk_batched():
    jvp_jaxpr = core.ClosedJaxpr(*jvp_jaxpr_thunk())
    jvp_in_batched = (*primals_in_batched, *primals_in_batched)
    # We need the primals and tangets to have the same batching. To work out
    # the appropriate instantiate pattern, batch the jaxpr once, then
    # use that output to decide on the pattern.
    instantiate = [d is not batching.not_mapped for d in prim_out_batched]
    instantiate = [*instantiate, *instantiate]
    _, out_batched = batching.batch_jaxpr(
        jvp_jaxpr, axis_size, jvp_in_batched, instantiate, axis_name,
        spmd_axis_name, main_type)
    primals_out_batched, tangents_out_batched = split_list(
        out_batched, [len(out_batched) // 2])
    out_batched = map(batching._merge_bdims, primals_out_batched,
                      tangents_out_batched)
    instantiate = [d is not batching.not_mapped for d in out_batched]
    batched_jaxpr, out_batched = batching.batch_jaxpr(
        jvp_jaxpr, axis_size, jvp_in_batched, instantiate, axis_name,
        spmd_axis_name, main_type)
    primals_out_batched, tangents_out_batched = split_list(
        out_batched, [len(out_batched) // 2])
    out_batched = map(batching._merge_bdims, primals_out_batched,
                      tangents_out_batched)
    out_dims2.append([0 if b else batching.not_mapped for b in out_batched])
    return batched_jaxpr.jaxpr, batched_jaxpr.consts

  # @pe._memoize
  def fwd_jaxpr_thunk_batched():
    assert 0, "TODO"

  @lu.wrap_init
  def bwd_batched(*args, **kwargs):
    assert 0, "TODO"

  @lu.wrap_init
  def lin_batched(*args, **kwargs):
    assert 0, "TODO"

  out_batched = custom_ad_p.bind(
      *args, num_consts=num_consts, name=name, prim_jaxpr=prim_jaxpr_batched,
      jvp_jaxpr_thunk=jvp_jaxpr_thunk_batched,
      fwd_jaxpr_thunk=fwd_jaxpr_thunk_batched if fwd_jaxpr_thunk else None,
      bwd=bwd_batched if bwd else None, lin=lin_batched if lin else None)

  out_dims = out_dims2[0] if out_dims2 else out_dims1
  return out_batched, out_dims


# def _custom_ad_staging(
#   trace,
#   *tracers,
#   num_consts: int,
#   name: str,
#   prim_jaxpr: core.ClosedJaxpr,
#   jvp: lu.WrappedFun | None,
#   fwd: lu.WrappedFun | None,
#   bwd: lu.WrappedFun | None,
#   lin: lu.WrappedFun | None,
# ):
#   main_ = ref(trace.main)
#   consts, tracers = split_list(tracers, [num_consts])
#   in_avals = [t.aval for t in tracers]

#   @pe._memoize
#   def jvp_jaxpr_thunk():
#     jaxpr, _, consts, atr = pe.trace_to_subjaxpr_dynamic(
#         jvp, main_(), in_avals + in_avals)
#     if atr: raise NotImplementedError
#     return jaxpr, consts

#   @pe._memoize
#   def fwd_jaxpr_thunk():
#     jaxpr, _, consts, atr = pe.trace_to_subjaxpr_dynamic(
#         fwd, main_(), in_avals)
#     if atr: raise NotImplementedError
#     return jaxpr, consts

#   out_avals = prim_jaxpr.out_avals
#   out_tracers = [pe.DynamicJaxprTracer(trace, x) for x in out_avals]
#   constvars = map(trace.getvar, map(trace.instantiate_const, consts))
#   invars = map(trace.getvar, tracers)
#   outvars = map(trace.makevar, out_tracers)
#   eqn = pe.new_jaxpr_eqn([*constvars, *invars], outvars, custom_ad_p,
#                          dict(num_consts=num_consts, name=name,
#                               prim_jaxpr=prim_jaxpr,
#                               jvp_jaxpr_thunk=jvp_jaxpr_thunk,
#                               fwd_jaxpr_thunk=fwd_jaxpr_thunk,
#                               bwd=bwd, lin=lin),
#                          prim_jaxpr.effects,
#                          source_info_util.current())
#   trace.frame.add_eqn(eqn)
#   return out_tracers


# class CustomAdPrimitive(core.Primitive):
#   multiple_results = True

#   def get_bind_params(self, params):
#     params = dict(params)
#     params["jvp"] = _lift_jaxpr_thunk(params.pop("jvp_jaxpr_thunk"))
#     params["fwd"] = _lift_jaxpr_thunk(params.pop("fwd_jaxpr_thunk"))
#     return [], params

# def _lift_jaxpr_thunk(jaxpr_thunk):
#   @lu.wrap_init
#   def jvp(*args):
#     jaxpr, consts = jaxpr_thunk()
#     return core.eval_jaxpr(jaxpr, consts, *args)
#   return jvp

custom_ad_p = core.Primitive("custom_ad")
custom_ad_p.multiple_results = True
custom_ad_p.def_impl(_custom_ad_impl)
custom_ad_p.def_abstract_eval(_custom_ad_abstract_eval)
ad.primitive_jvps[custom_ad_p] = _custom_ad_jvp
ad.primitive_transposes[custom_ad_p] = _custom_ad_transpose
mlir.register_lowering(
    custom_ad_p, mlir.lower_fun(_custom_ad_impl, multiple_results=True))
# pe.custom_staging_rules[custom_ad_p] = _custom_ad_staging
batching.spmd_axis_primitive_batchers[custom_ad_p] = _custom_ad_batching
batching.axis_primitive_batchers[custom_ad_p] = partial(
    _custom_ad_batching, None)


def _jvp_helper_impl(*args, num_consts: int, jvp_jaxpr_thunk, **_):
  _, args = split_list(args, [num_consts])
  jaxpr, consts = jvp_jaxpr_thunk()
  return core.eval_jaxpr(jaxpr, consts, *args)

def _jvp_helper_abstract_eval(*args, prim_jaxpr: core.ClosedJaxpr, **_):
  del args  # unused
  out_avals = prim_jaxpr.out_avals
  return (*out_avals, *out_avals)

def _jvp_helper_partial_eval(
    trace, *tracers, num_consts: int, name: str, prim_jaxpr: core.ClosedJaxpr,
    jvp_jaxpr_thunk: Callable, fwd_jaxpr_thunk: Callable | None,
    bwd: lu.WrappedFun | None, lin: lu.WrappedFun | None):
  assert fwd_jaxpr_thunk is not None
  consts, tracers = split_list(tracers, [num_consts])
  assert all(t.pval.is_known() for t in consts)
  primals, tangents = split_list(tracers, [len(tracers) // 2])
  assert all(t.pval.is_known() for t in primals)
  tangents = [trace.instantiate_const(t) if t.pval.is_known() else t
              for t in tangents]
  assert all(not t.pval.is_known() for t in tangents)
  out_avals = prim_jaxpr.out_avals

  knowns = [t.pval.get_known() for t in primals]
  primals_out_and_res = core.eval_jaxpr(*fwd_jaxpr_thunk(), *knowns)
  res, primals_out = split_list(
      primals_out_and_res, [len(primals_out_and_res) - len(out_avals)])

  res = map(trace.new_instantiated_const, res)
  consts = map(trace.instantiate_const, consts)
  primals = map(trace.instantiate_const, primals)
  tangents_out = [
    pe.JaxprTracer(trace, pe.PartialVal.unknown(a), None) for a in out_avals
  ]
  name_stack = source_info_util.current_name_stack()[len(trace.name_stack) :]
  source = source_info_util.current().replace(name_stack=name_stack)
  params = dict(num_res=len(res), num_consts=num_consts, name=name,
                prim_jaxpr=prim_jaxpr, jvp_jaxpr_thunk=jvp_jaxpr_thunk,
                fwd_jaxpr_thunk=fwd_jaxpr_thunk, bwd=bwd, lin=lin)
  eqn = pe.new_eqn_recipe(
      [*res, *consts, *primals, *tangents], tangents_out, transpose_helper_p,
      params, prim_jaxpr.effects, source)
  for t in tangents_out:
    t.recipe = eqn

  return primals_out + tangents_out


# def _jvp_helper_staging(
#   trace,
#   *tracers,
#   num_consts: int,
#   name: str,
#   prim_jaxpr: core.ClosedJaxpr,
#   jvp: lu.WrappedFun | None,
#   fwd: lu.WrappedFun | None,
#   bwd: lu.WrappedFun | None,
#   lin: lu.WrappedFun | None,
# ):
#   main_ = ref(trace.main)
#   consts, tracers = split_list(tracers, [num_consts])
#   primals, tangents = split_list(tracers, [len(tracers) // 2])
#   primal_avals = [t.aval for t in primals]
#   tangent_avals = [t.aval for t in tangents]

#   @pe._memoize
#   def jvp_jaxpr_thunk():
#     jaxpr, _, consts, atr = pe.trace_to_subjaxpr_dynamic(
#         jvp, main_(), primal_avals + tangent_avals)
#     if atr: raise NotImplementedError
#     return jaxpr, consts

#   @pe._memoize
#   def fwd_jaxpr_thunk():
#     jaxpr, _, consts, atr = pe.trace_to_subjaxpr_dynamic(
#         fwd, main_(), primal_avals)
#     if atr: raise NotImplementedError
#     return jaxpr, consts

#   out_avals = prim_jaxpr.out_avals + prim_jaxpr.out_avals
#   out_tracers = [pe.DynamicJaxprTracer(trace, x) for x in out_avals]
#   constvars = map(trace.getvar, map(trace.instantiate_const, consts))
#   invars = map(trace.getvar, tracers)
#   outvars = map(trace.makevar, out_tracers)
#   eqn = pe.new_jaxpr_eqn([*constvars, *invars], outvars, jvp_helper_p,
#                          dict(num_consts=num_consts, name=name,
#                              prim_jaxpr=prim_jaxpr,
#                              jvp_jaxpr_thunk=jvp_jaxpr_thunk,
#                              fwd_jaxpr_thunk=fwd_jaxpr_thunk,
#                              bwd=bwd, lin=lin),
#                          prim_jaxpr.effects,
#                          source_info_util.current())
#   trace.frame.add_eqn(eqn)
#   return out_tracers


def _jvp_helper_dce(used_outputs: list[bool], eqn: core.JaxprEqn):
  primals_out_used, tangents_out_used = split_list(
      used_outputs, [len(used_outputs) // 2])
  if any(tangents_out_used):
    return [True] * len(eqn.invars), eqn
  prim_jaxpr = eqn.params["prim_jaxpr"]
  prim_jaxpr, primals_in_used = pe.dce_jaxpr(prim_jaxpr.jaxpr, primals_out_used)
  prim_jaxpr = pe.close_jaxpr(prim_jaxpr)
  invars = eqn.invars[:len(primals_in_used)]
  invars = [v for v, used in zip(invars, primals_in_used) if used]
  outvars = eqn.outvars[:len(primals_out_used)]
  outvars = [v for v, used in zip(outvars, primals_out_used) if used]
  new_eqn = pe.new_jaxpr_eqn(invars, outvars, custom_ad_p, eqn.params,
                             prim_jaxpr.effects, eqn.source_info)
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
# pe.custom_staging_rules[jvp_helper_p] = _jvp_helper_staging
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
  del name, prim_jaxpr, jvp, fwd, bwd
  assert lin is not None
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
