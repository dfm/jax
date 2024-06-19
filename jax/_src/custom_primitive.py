from typing import Callable

from jax._src import core
from jax._src import linear_util as lu
from jax._src.custom_derivatives import (_resolve_kwargs, _flatten_fun_nokwargs,
                                         _flatten_jvp)
from jax._src.interpreters import mlir
from jax._src.tree_util import tree_flatten, tree_unflatten
from jax._src.util import safe_map

map = safe_map

def build_custom_primitive(namespace, *, name: str | None = None) -> Callable:
  if name is None:
    name = getattr(namespace, "__name__", str(namespace))

  if not hasattr(namespace, "impl"):
    raise TypeError("A custom primitive requires an `impl`.")

  prim = CustomPrimitive(name)
  mlir.register_lowering(prim, _custom_primitive_lowering)

  def call_custom_primitive(*args, **kwargs):
    args = _resolve_kwargs(namespace.impl, args, kwargs)
    fun = lu.wrap_init(namespace.impl)
    args_flat, in_tree = tree_flatten(args)

    fun_flat, fun_out_type = _flatten_fun_nokwargs(fun, in_tree)

    def _jvp(*args, **kwargs):
      raise NotImplementedError()
    jvp = getattr(namespace, "jvp", _jvp)
    jvp_flat, jvp_out_type = _flatten_jvp(lu.wrap_init(jvp), name,
                                          f"{name}_jvp", in_tree, fun_out_type)

    out_flat = prim.bind(fun_flat, jvp_flat, *args_flat)

    if jvp_flat is not None:
      _, (out_tree, _) = lu.merge_linear_aux(fun_out_type, jvp_out_type)
    else:
      out_tree, _ = fun_out_type()

    return tree_unflatten(out_tree, out_flat)

  return call_custom_primitive


class CustomPrimitive(core.Primitive):
  multiple_results = True

  def bind(self, fun, jvp, *args):
    args = map(core.full_lower, args)
    top_trace = core.find_top_trace(args)
    fun, fun_todo = process_env_traces(fun, self, top_trace and top_trace.level)
    jvp, jvp_todo = process_env_traces(jvp, self, top_trace and top_trace.level)
    tracers = map(top_trace.full_raise, args)
    outs = top_trace.process_custom_primitive(self, tracers, fun, jvp)
    _, env_trace_todo = lu.merge_linear_aux(fun_todo, jvp_todo)
    return core.apply_todos(env_trace_todo, map(core.full_lower, outs))

  def impl(self, fun, jvp, *args):
    del jvp
    with core.new_sublevel():
      return fun.call_wrapped(*args)

  def post_process(self, trace, out_tracers):
    return trace.post_process_custom_primitive(out_tracers)

  def get_bind_params(self, params):
    new_params = dict(params)
    call_jaxpr = new_params.pop("call_jaxpr")
    num_consts = new_params.pop("num_consts")
    # jvp_jaxpr_thunk = new_params.pop('jvp_jaxpr_thunk')
    fun = lu.wrap_init(core.jaxpr_as_fun(call_jaxpr))
    # jvp = lift_jvp(num_consts, jvp_jaxpr_thunk)
    return [fun, None], new_params


@lu.transformation_with_aux
def process_env_traces(primitive, level: int, *args):
  outs = yield args, {}
  todo = []
  while True:
    tracers = [x for x in outs if isinstance(x, core.Tracer)
               and (level is None or x._trace.level > level)]
    if tracers:
      ans = max(tracers, key=lambda x: x._trace.level)
    else:
      break
    trace = ans._trace.main.with_cur_sublevel()
    outs = map(trace.full_raise, outs)
    outs, cur_todo = primitive.post_process(trace, outs)
    todo.append(cur_todo)
  yield outs, tuple(todo)


def _custom_primitive_lowering(ctx, *args, call_jaxpr, num_consts):
  args_ = map(mlir.wrap_singleton_ir_values, args)
  consts = mlir._ir_consts(call_jaxpr.consts)
  out, tokens = mlir.jaxpr_subcomp(ctx.module_context, call_jaxpr.jaxpr,
                                   ctx.name_stack, ctx.tokens_in, consts,
                                   *args_, dim_var_values=ctx.dim_var_values)
  ctx.set_tokens_out(tokens)
  return out
