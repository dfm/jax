import functools
from typing import Callable

from jax._src import api_util
from jax._src import core
from jax._src import linear_util as lu
from jax._src.interpreters import ad
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.tree_util import tree_flatten, tree_unflatten
from jax._src.util import safe_map, split_list

map = safe_map


def build_custom_primitive(spec, *, name: str | None = None) -> Callable:
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
  ad.primitive_jvps[prim] = custom_primitive_jvp

  return prim


class CustomPrimitive(core.Primitive):
  multiple_results = True

  def __init__(self, name: str, spec):
    super().__init__(name)
    functools.update_wrapper(self, spec)
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

    out_flat = self.bind(*consts, *args_flat, call_jaxpr=call_jaxpr,
                         num_consts=len(consts), in_tree=in_tree,
                         out_tree=out_tree, name=self.name, spec=self.spec,
                         kwargs=kwargs)

    return tree_unflatten(out_tree, out_flat)


def custom_primitive_impl(*args, call_jaxpr: core.ClosedJaxpr, **_):
  return core.jaxpr_as_fun(call_jaxpr)(*args)


def custom_primitive_abstract_eval(*args, call_jaxpr: core.ClosedJaxpr, **_):
  del args
  return call_jaxpr.out_avals, call_jaxpr.effects


def custom_primitive_lowering(ctx, *args, call_jaxpr: core.ClosedJaxpr, **_):
  args_ = map(mlir.wrap_singleton_ir_values, args)
  consts = mlir._ir_consts(call_jaxpr.consts)
  out, tokens = mlir.jaxpr_subcomp(ctx.module_context, call_jaxpr.jaxpr,
                                   ctx.name_stack, ctx.tokens_in, consts,
                                   *args_, dim_var_values=ctx.dim_var_values)
  ctx.set_tokens_out(tokens)
  return out


def custom_primitive_jvp(primals, tangents, call_jaxpr: core.ClosedJaxpr,
                         num_consts: int, in_tree, out_tree, name: str, spec,
                         kwargs):
  del call_jaxpr

  if hasattr(spec, "vjp_fwd"):
    assert hasattr(spec, "vjp_bwd")
    # TODO(dfm): Add vjp support
    raise NotImplementedError("todo support vjp")

  if not hasattr(spec, "jvp"):
    raise NotImplementedError(f"'jvp' not implemented for custom primitive '{name}'")

  _, primals = split_list(primals, [num_consts])
  const_tangents, tangents = split_list(tangents, [num_consts])
  assert all(isinstance(t, ad.Zero) for t in const_tangents)

  py_primals = tree_unflatten(in_tree, primals)
  py_tangents = tree_unflatten(in_tree, tangents)
  py_primals_out, py_tangents_out = spec.jvp(py_primals, py_tangents, **kwargs)

  primals_out, out_tree1 = tree_flatten(py_primals_out)
  tangents_out, out_tree2 = tree_flatten(py_tangents_out)
  assert out_tree1 == out_tree, "todo error"
  assert out_tree2 == out_tree, "todo error"

  return primals_out, tangents_out
