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

from collections.abc import Sequence, Callable
from functools import partial, update_wrapper
from typing import Any, Generic, TypeVar

from jax._src import ad_util
from jax._src import api_util
from jax._src import core
from jax._src import linear_util as lu
from jax._src import traceback_util
from jax._src.errors import UnexpectedTracerError
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import partial_eval as pe
from jax._src.tree_util import (tree_flatten, tree_unflatten, treedef_children,
                                tree_map, tree_flatten_with_path, tree_leaves)
from jax._src.util import safe_map, safe_zip, split_list, unzip2, Unhashable

map = safe_map
zip = safe_zip

ReturnT = TypeVar("ReturnT")
T = TypeVar("T")


class custom_primitive(core.Primitive, Generic[ReturnT]):
  fun: Callable[..., ReturnT]
  static_argnums: Sequence[int]
  multiple_results = True

  # The following are the hooks that can be customized using this API.
  vmap: Callable[..., ReturnT] | None = None
  jvp: Callable[..., tuple[ReturnT, Any]] | None = None
  fwd: Callable[..., tuple[ReturnT, Any]] | None = None
  bwd: Callable[..., Any] | None = None
  lin: Callable[..., Any] | None = None

  def __init__(self, fun: Callable[..., ReturnT], *, name: str | None = None,
               static_argnums: Sequence[int] = ()) -> None:
    update_wrapper(self, fun)
    self.fun = fun
    self.static_argnums = static_argnums
    if name is None:
      name = getattr(fun, "__name__", str(fun))
    super().__init__(name)

    self.def_impl(_custom_primitive_impl)
    self.def_abstract_eval(_custom_primitive_abstract_eval)
    ad.primitive_jvps[self] = _custom_primitive_jvp
    batching.spmd_axis_primitive_batchers[self] = _custom_primitive_batching
    batching.axis_primitive_batchers[self] = partial(
        _custom_primitive_batching, None)


  def defvmap(self, fun: Callable[..., ReturnT]) -> Callable[..., ReturnT]:
    self.vmap = fun
    return fun

  def defjvp(
      self,
      jvp: Callable[..., tuple[ReturnT, T]],
  ) -> Callable[..., tuple[ReturnT, T]]:
    self.jvp = jvp
    return jvp

  def deffwd(
      self,
      fwd: Callable[..., tuple[ReturnT, T]],
  ) -> Callable[..., tuple[ReturnT, T]]:
    self.fwd = fwd
    return fwd

  def defbwd(self, bwd: Callable[..., T]) -> Callable[..., T]:
    self.bwd = bwd
    return bwd

  def deflin(self, lin: Callable[..., T]) -> Callable[..., T]:
    self.lin = lin
    return lin

  def defvjp(
      self,
      fwd: Callable[..., tuple[ReturnT, Any]],
      bwd: Callable[..., Any],
  ) -> None:
    self.fwd = fwd
    self.bwd = bwd

  @traceback_util.api_boundary
  def __call__(self, *args: Any, **kwargs: Any) -> ReturnT:
    args = api_util.resolve_kwargs(self.fun, args, kwargs)

    if self.static_argnums:
      for i in self.static_argnums:
        _check_for_tracers(args[i])
      static_argnums = set(self.static_argnums)
      dyn_argnums = [i for i in range(len(args)) if i not in static_argnums]
      fun, dyn_args = api_util.argnums_partial(
          lu.wrap_init(self.fun), dyn_argnums, args,
          require_static_args_hashable=False)
      static_args = [args[i] for i in sorted(self.static_argnums)]
    else:
      fun, dyn_args = lu.wrap_init(self.fun), args
      static_args = []

    # Since this is an "initial style" primitive, we trace the primal function
    # to a jaxpr right off the bat. This should be updated eventually, but for
    # now it significantly simplifies the implementation.
    args_flat, tree_in = tree_flatten(dyn_args)
    fun_flat, tree_out_thunk = api_util.flatten_fun_nokwargs(fun, tree_in)
    avals_in = [core.raise_to_shaped(core.get_aval(x)) for x in args_flat]
    debug = pe.debug_info(fun, tree_in, tree_out_thunk, False,
                          "custom_primitive")
    jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(fun_flat, avals_in, debug)
    prim_jaxpr = pe.close_jaxpr(pe.convert_constvars_jaxpr(jaxpr))
    tree_out = tree_out_thunk()

    if self.vmap is not None:
      vmap = self.vmap
      rule_name = getattr(vmap, "__name__", str(vmap))
      vmap = _add_args(lu.wrap_init(vmap), static_args)
      vmap = _flatten_vmap(vmap, self.name, rule_name, tree_in, tree_out)
    else:
      vmap = None

    if self.jvp is not None:
      jvp = self.jvp
      rule_name = getattr(jvp, "__name__", str(jvp))
      jvp = _add_args(lu.wrap_init(jvp), static_args)
      jvp = _flatten_jvp(jvp, self.name, rule_name, tree_in, tree_out,
                         prim_jaxpr.out_avals)

      @pe._memoize
      def jvp_jaxpr_thunk():
        jvp_avals_in = (*avals_in, *avals_in)
        jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(jvp, jvp_avals_in)
        return jaxpr, consts
    else:
      jvp_jaxpr_thunk = None

    if self.fwd is not None:
      fwd = self.fwd
      rule_name = getattr(fwd, "__name__", str(fwd))
      fwd = _add_args(lu.wrap_init(fwd), static_args)
      fwd, res_type_thunk = _flatten_fwd(fwd, self.name, rule_name, tree_in,
                                         tree_out, prim_jaxpr.out_avals)

      @pe._memoize
      def fwd_jaxpr_thunk():
        jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(fwd, avals_in)
        return jaxpr, consts

    else:
      fwd_jaxpr_thunk = None
      res_type_thunk = None

    bwd = self.bwd
    if bwd:
      if fwd is None:
        raise AttributeError(
            f"No fwd rule defined for the custom primitive {self.name}. When a "
            "bwd rule is defined, a fwd rule must also be set using deffwd."
        )
      bwd_name = getattr(bwd, "__name__", str(bwd))
      bwd = _add_args(lu.wrap_init(bwd), static_args)
      bwd = _flatten_bwd(bwd, self.name, bwd_name, avals_in, tree_in, tree_out,
                         res_type_thunk)

    lin = self.lin
    if lin:
      if fwd is None:
        raise AttributeError(
            f"No fwd rule defined for the custom primitive {self.name}. When a "
            "lin rule is defined, a fwd rule must also be set using deffwd."
        )
      rule_name = getattr(lin, "__name__", str(lin))
      lin = _add_args(lu.wrap_init(lin), static_args)
      lin = _flatten_lin(lin, self.name, rule_name, tree_in, tree_out,
                         prim_jaxpr.out_avals, res_type_thunk)

    out_flat = self.bind(
        *consts, *args_flat, name=self.name, num_consts=len(consts),
        prim_jaxpr=prim_jaxpr, vmap=vmap, jvp_jaxpr_thunk=jvp_jaxpr_thunk,
        fwd_jaxpr_thunk=fwd_jaxpr_thunk, bwd=bwd, lin=lin)
    return tree_unflatten(tree_out, out_flat)


def _check_for_tracers(x):
  for leaf in tree_leaves(x):
    if isinstance(leaf, core.Tracer):
      raise UnexpectedTracerError(
          "Found a JAX Tracer object passed as an argument to a custom "
          "primitive in a position indicated by static_argnums as static. "
          "Tracers cannot be passed as static arguments to custom primitives; "
          "instead, static_argnums should only be used for arguments that "
          "can't be or contain JAX tracers, e.g. function-valued arguments. "
          "In particular, array-valued arguments should typically not be "
          "indicated as static_argnums."
      )


def _add_args(f, extra_args):
  if extra_args:
    return _add_args_(f, tuple(Unhashable(arg) for arg in extra_args))
  else:
    return f


@lu.transformation
def _add_args_(extra_args, *args, **kwargs):
  extra_args = tuple(arg.val for arg in extra_args)
  all_args = (extra_args + args)
  yield (yield all_args, kwargs)


@lu.transformation
def _flatten_vmap(name, rule_name, tree_in, tree_out, axis_size, *args):
  batched_in, args = split_list(args, [len(args) // 2])
  py_batched_in = tree_unflatten(tree_in, batched_in)
  py_args = tree_unflatten(tree_in, args)
  pair_out = yield (axis_size, py_batched_in, *py_args), {}
  if not isinstance(pair_out, (list, tuple)) or len(pair_out) != 2:
    raise TypeError(
        f"The vmap rule {rule_name} for custom primitive {name} must produce a "
        "pair (list or tuple of length two) representing the batched outputs "
        f"and the output batching pattern, but got {pair_out}."
    )
  py_out, py_batched_out = pair_out
  out, tree_out_ = tree_flatten(py_out)
  batched_out, tree_batched_out = tree_flatten(py_batched_out)
  if tree_out_ != tree_batched_out:
    raise TypeError(
        f"The vmap rule {rule_name} for custom primitive {name} must produce "
        "value and batching outputs with equal pytree structures, but got "
        f"{tree_out_} and {tree_batched_out} respectively."
    )
  if tree_out_ != tree_out:
    raise TypeError(
        f"The vmap rule {rule_name} for custom primitive {name} must produce a "
        "pair (list or tuple of length two) where the first element represents "
        "the batched outputs. This output must have the same pytree structure "
        f"as the primal, but got {tree_out_} and {tree_out} respectively."
    )
  yield (*out, *batched_out)

@lu.transformation
def _flatten_jvp(name, rule_name, tree_in, tree_out, avals_out, *args):
  primals_in, tangents_in = split_list(args, [len(args) // 2])
  py_primals = tree_unflatten(tree_in, primals_in)
  py_tangents = tree_unflatten(tree_in, tangents_in)
  pair_out = yield (py_primals, py_tangents), {}
  if not isinstance(pair_out, (list, tuple)) or len(pair_out) != 2:
    raise TypeError(
        f"The jvp rule {rule_name} for custom primitive {name} must produce a "
        "pair (list or tuple of length two) representing primal and tangent "
        f"outputs, but got {pair_out}."
    )
  py_primals_out, py_tangents_out = pair_out
  primals_out, tree_prim = tree_flatten(py_primals_out)
  tangents_out, tree_tan = tree_flatten(py_tangents_out)
  if tree_prim != tree_tan:
    raise TypeError(
        f"The jvp rule {rule_name} for custom primitive {name} must produce "
        "primal and tangent outputs with equal pytree structures, but got "
        f"{tree_prim} and {tree_tan} respectively."
    )
  avals_prim = [core.raise_to_shaped(core.get_aval(x)) for x in primals_out]
  prim_ty_tree = tree_unflatten(tree_prim, [a.str_short() for a in avals_prim])
  orig_ty_tree = tree_unflatten(tree_out, [a.str_short() for a in avals_out])
  if tree_prim != tree_out:
    raise TypeError(
        f"The jvp rule {rule_name} for custom primitive {name} must produce a "
        "pair (list or tuple of length two) where the first element represents "
        "the primal output. This output must have the same pytree structure as "
        "the primal, but the output's first element had pytree structure:\n"
        f"""    {str(prim_ty_tree).replace("'", "")}\n"""
        "while the primal function had output pytree structure:\n"
        f"""    {str(orig_ty_tree).replace("'", "")}."""
    )
  if not all(map(core.typematch, avals_prim, avals_out)):
    raise TypeError(
        f"The jvp rule {rule_name} for custom primitive {name} must produce a "
        "pair (list or tuple of length two) where the first element represents "
        "the primal output. This output must have leaves of the same shape and "
        "dtype as the primal, but the output's first element had the following "
        "shapes and dtypes:\n"
        f"""    {str(prim_ty_tree).replace("'", "")}\n"""
        "while the primal function had the following output shapes and dtypes:\n"
        f"""    {str(orig_ty_tree).replace("'", "")}"""
    )
  primal_avals_out = [core.raise_to_shaped(core.get_aval(x), weak_type=False)
                      for x in primals_out]
  tangent_avals_out = [core.raise_to_shaped(core.get_aval(t), weak_type=False)
                       for t in tangents_out]
  if primal_avals_out != tangent_avals_out:
    disagreements = "\n".join(
        f"  primal {av1.str_short()} for tangent {av2.str_short()}"
        for av1, av2 in zip(primal_avals_out, tangent_avals_out) if av1 != av2)
    raise TypeError(
        f"The jvp rule {rule_name} for custom primitive {name} must produce "
        "primal and tangent outputs with equal shapes and dtypes, but "
        f"{rule_name} returned:\n{disagreements}"
    )
  yield (*primals_out, *tangents_out)


@lu.transformation_with_aux
def _flatten_fwd(name, rule_name, tree_in, tree_out, avals_out, *args):
  py_args = tree_unflatten(tree_in, args)
  pair_out = yield py_args, {}
  if not isinstance(pair_out, (list, tuple)) or len(pair_out) != 2:
    raise TypeError(
        f"The fwd rule {rule_name} for custom primitive {name} must produce a "
        "pair (list or tuple of length two) where the first element represents "
        "the primal output and the second element represents residuals, which "
        "are saved from the forward pass and used as input to the bwd and lin "
        f"rules. Instead, the fwd rule {rule_name} produced {pair_out}."
    )
  py_primals_out, py_res = pair_out
  primals_out, tree_prim = tree_flatten(py_primals_out)
  avals_prim = [core.raise_to_shaped(core.get_aval(x)) for x in primals_out]
  fwd_ty_tree = tree_unflatten(tree_out, [a.str_short() for a in avals_prim])
  prim_ty_tree = tree_unflatten(tree_out, [a.str_short() for a in avals_out])
  if tree_prim != tree_out:
    raise TypeError(
        f"The fwd rule {rule_name} for custom primitive {name} must produce a "
        "pair (list or tuple of length two) where the first element represents "
        "the primal output. This output must have the same pytree structure as "
        "the primal, but the output's first element had pytree structure:\n"
        f"""    {str(fwd_ty_tree).replace("'", "")}\n"""
        f"while the function {name} had output pytree structure:\n"
        f"""    {str(prim_ty_tree).replace("'", "")}."""
    )
  if not all(map(core.typematch, avals_prim, avals_out)):
    raise TypeError(
        f"The fwd rule {rule_name} for custom primitive {name} must produce a "
        "pair (list or tuple of length two) where the first element represents "
        "the primal output. This output must have leaves of the same shape and "
        "dtype as the primal, but the output's first element had the following "
        "shapes and dtypes:\n"
        f"""    {str(fwd_ty_tree).replace("'", "")}\n"""
        "while the primal function had the following output shapes and dtypes:\n"
        f"""    {str(prim_ty_tree).replace("'", "")}\n"""
    )
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
    raise TypeError(
        f"The bwd rule {rule_name} for custom primitive {name} must produce an "
        "output with the same pytree structure as the args tuple of the primal "
        "function, and in particular must produce a tuple of length equal to "
        "the number of arguments to the primal function. Instead, the bwd rule "
        f"{rule_name} produced an output with structure {tree_in2}, whereas "
        f"the primal input structure was {tree_in}."
    ) from None
  results = []
  for kp, a, ct in zip(keypaths, avals_in, cts_in_flat):
    del kp  # unused
    if ct is zero or a != a.at_least_vspace():
      results.append(ad_util.Zero(a.at_least_vspace()))
    else:
      # TODO(dfm): What's going on here? This was copied from custom_vjp.
      # if (not core.typecompat(a.at_least_vspace(), a_ := core.get_aval(ct))
      #     and not (cd._temporary_dtype_exception(a, a_) or
      #              cd._temporary_shape_exception(a, a_))):
      #   msg = (
      #     "Custom VJP bwd rule must produce an output with the same "
      #     f"shape/dtypes as the args tuple of the primal function {name}, but "
      #     f"at output{keystr(kp)} the bwd rule produced an output of "
      #     f"shape/dtype {core.raise_to_shaped(a_).str_short()} corresponding "
      #     f"to an input of shape/dtype {a.str_short()}.")
      #   raise ValueError(msg)
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
    raise TypeError(
        f"The lin rule {rule_name} for custom primitive {name} must produce "
        "tangent outputs with the same pytree structure as the primal "
        f"function, but got {tree_tan} and {tree_out} respectively."
    )
  tangent_avals_out = [core.raise_to_shaped(core.get_aval(t), weak_type=False)
                       for t in tangents_out]
  if avals_out != tangent_avals_out:
    disagreements = "\n".join(
        f"  primal {av1.str_short()} for tangent {av2.str_short()}"
        for av1, av2 in zip(avals_out, tangent_avals_out) if av1 != av2)
    raise TypeError(
        f"The lin rule {rule_name} for custom primitive {name} must produce "
        "primal and tangent outputs with equal shapes and dtypes, but "
        f"{rule_name} returned:\n{disagreements}"
    )
  yield tangents_out


def _custom_primitive_impl(*args, prim_jaxpr: core.ClosedJaxpr, **_):
  return core.jaxpr_as_fun(prim_jaxpr)(*args)


def _custom_primitive_abstract_eval(*args, prim_jaxpr: core.ClosedJaxpr, **_):
  del args  # unused
  return prim_jaxpr.out_avals


class CustomPrimitiveBatchingException(Exception):
  pass


def _custom_primitive_batching(
    spmd_axis_name, axis_size, axis_name, main_type, args, dims,
    num_consts: int, name: str, vmap: lu.WrappedFun | None, **_,
):
  del spmd_axis_name, axis_name, main_type  # unused
  if vmap is None:
    raise CustomPrimitiveBatchingException(
        f"The custom primitive {name} does not have a vmap rule defined"
    )
  if any(d is not batching.not_mapped for d in dims[:num_consts]):
    raise CustomPrimitiveBatchingException(
        f"Detected batchgin of custom primitive {name} with respect to a "
        "closed-over value. This is not supported because the custom vmap rule "
        "only specifies how to batch with respect to the explicit input "
        "arguments. Try passing the closed-over value as an argument and "
        "updating the batching rule."
    )
  _, args = split_list(args, [num_consts])
  _, dims = split_list(dims, [num_consts])
  axis_size, = {x.shape[d] for x, d in zip(args, dims)
                if d is not batching.not_mapped}
  args = [batching.moveaxis(x, d, 0) if d is not batching.not_mapped and d != 0
          else x for x, d in zip(args, dims)]
  in_batched = [d is not batching.not_mapped for d in dims]
  out = vmap.call_wrapped(axis_size, *in_batched, *args)
  flat_out, out_batched = split_list(out, [len(out) // 2])
  flat_out_dims = [0 if d else batching.not_mapped for d in out_batched]
  return flat_out, flat_out_dims


class CustomPrimitiveADException(Exception):
  pass


def _custom_primitive_jvp(
    primals, tangents, *, name: str, num_consts: int,
    prim_jaxpr: core.ClosedJaxpr, vmap: lu.WrappedFun | None,
    jvp_jaxpr_thunk: Callable | None, fwd_jaxpr_thunk: Callable | None,
    bwd: lu.WrappedFun | None, lin: lu.WrappedFun | None
):
  if any(not isinstance(t, ad_util.Zero) for t in tangents[:num_consts]):
    raise CustomPrimitiveADException(
        f"Detected differentiation of custom primitive {name} with respect to "
        "a closed-over value. This is not supported because the custom "
        "differentiation rules only specifiy how to differentiate with respect "
        "to the explicit input arguments. Try passing the closed-over value as "
        "an argument and updating the differentiation rules."
    )
  consts, primals_in = split_list(primals, [num_consts])
  _, tangents_in = split_list(tangents, [num_consts])
  tangents_in = map(ad.instantiate_zeros, tangents_in)
  if all(f is None for f in (jvp_jaxpr_thunk, bwd, lin)):
    raise CustomPrimitiveADException(
        f"The custom primitive {name} cannot be differentiated without an "
        "explicit custom differentiation rule. Please define at least one of "
        f"jvp, bwd, or lin for the {name} custom primitive."
    )
  if lin is None and bwd is None:
    assert jvp_jaxpr_thunk is not None
    jvp_jaxpr, jvp_consts = jvp_jaxpr_thunk()
    closed_jvp_jaxpr = pe.close_jaxpr(pe.convert_constvars_jaxpr(jvp_jaxpr))
    out_flat = core.call(lu.wrap_init(core.jaxpr_as_fun(closed_jvp_jaxpr)),
                         *jvp_consts, *primals_in, *tangents_in)
  else:
    out_flat = custom_primitive_deferred_fwd_p.bind(
        *consts, *primals_in, *tangents_in, num_consts=num_consts, name=name,
        prim_jaxpr=prim_jaxpr, vmap=vmap, jvp_jaxpr_thunk=jvp_jaxpr_thunk,
        fwd_jaxpr_thunk=fwd_jaxpr_thunk, bwd=bwd, lin=lin)
  primals_out, tangents_out = split_list(out_flat, [len(out_flat) // 2])
  return primals_out, tangents_out
