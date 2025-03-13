from collections.abc import Callable, Sequence
import dataclasses
from functools import wraps

from jax._src import api_util
from jax._src import core
from jax._src import linear_util as lu
from jax._src import util
from jax._src import source_info_util
from jax._src import tree_util
from jax._src.interpreters import ad
from jax._src.interpreters import partial_eval as pe

map = util.safe_map
zip = util.safe_zip


def out_of_line(fun: Callable, *, static_argnums: Sequence[int] = ()):
  static_argnums = set(static_argnums)

  @wraps(fun)
  def wrapped(*args, **kwargs):
    debug_info = api_util.debug_info("out_of_line", fun, args, kwargs)
    args = api_util.resolve_kwargs(fun, args, kwargs)

    if static_argnums:
      dyn_argnums = [i for i in range(len(args)) if i not in static_argnums]
      static_args = tuple(x for i, x in enumerate(args) if i in static_argnums)
      f_, dyn_args = api_util.argnums_partial(
          lu.wrap_init(fun, debug_info=debug_info), dyn_argnums, args,
          require_static_args_hashable=False)
    else:
      f_, dyn_args = lu.wrap_init(fun, debug_info=debug_info), args
      static_args = ()

    args_flat, in_tree = tree_util.tree_flatten(dyn_args)
    flat_fun, out_tree = api_util.flatten_fun_nokwargs(f_, in_tree)
    in_avals = tuple(core.get_aval(x) for x in args_flat)
    key = core.JaxprModuleKey(
        fun=fun, static_argnums=tuple(static_argnums), static_args=static_args,
        in_avals=in_avals, transformation_stack=())
    out_flat = out_of_line_call_p.bind(flat_fun, *args_flat, in_tree=in_tree,
                                       key=key, num_consts=0)
    return tree_util.tree_unflatten(out_tree(), out_flat)

  return wrapped


class OutOfLineCallPrimitive(core.CallPrimitive):
  def get_bind_params(self, params, module: core.JaxprModule | None = None):
    assert module is not None
    key = params["key"]
    jaxpr, *_ = module[key]
    subfun = lu.hashable_partial(
        lu.wrap_init(core.eval_jaxpr, debug_info=jaxpr.debug_info), jaxpr, ())
    # if config.dynamic_shapes.value:
    #   subfun = lu.annotate(subfun, _jaxpr_type_to_callable_annotation(jaxpr))
    return [subfun], params

  def bind_with_trace(self, trace, fun_and_args, params):
    fun = fun_and_args[0]
    args = fun_and_args[1:]
    return trace.process_out_of_line_call(self, fun, args, params)


out_of_line_call_p = OutOfLineCallPrimitive("out_of_line_call")
out_of_line_call_p.def_impl(core.call_impl)


def djt_process_out_of_line_call(
    trace: pe.DynamicJaxprTrace, primitive: OutOfLineCallPrimitive,
    f: lu.WrappedFun, explicit_tracers, params):
  if f.in_type is None:
    f = lu.annotate(f, tuple((core.get_aval(t), True) for t in explicit_tracers))
  assert f.in_type is not None
  implicit_tracers = pe._extract_implicit_args(trace, f.in_type, explicit_tracers)
  in_tracers = map(trace.to_jaxpr_tracer, [*implicit_tracers, *explicit_tracers])

  key = params["key"]
  if key in trace.frame.module:
    jaxpr, out_type, consts, stores = trace.frame.module[key]
    f.populate_stores(stores)
  else:
    jaxpr, out_type, consts = pe.trace_to_jaxpr_dynamic2(f)
    trace.frame.module[key] = core.JaxprModuleValue(jaxpr, out_type, consts, f.stores)

  # TODO(dfm): Probably need to handle dynamic shapes here...

  source_info = source_info_util.current()
  out_tracers = [pe.DynamicJaxprTracer(trace, aval, source_info)
                 for aval, _ in out_type]

  invars = map(trace.getvar, in_tracers)
  constvars = map(trace.getvar, map(trace.to_jaxpr_tracer, consts))
  outvars = map(trace.makevar, out_tracers)
  new_params = dict(params)
  new_params["num_consts"] += len(consts)
  eqn = pe.new_jaxpr_eqn(
      [*constvars, *invars], outvars, primitive,  new_params, jaxpr.effects,
      source_info)
  trace.frame.add_eqn(eqn)
  return [t for t, (_, keep) in zip(out_tracers, out_type) if keep]

pe.DynamicJaxprTrace.process_out_of_line_call = djt_process_out_of_line_call

def jvp_process_out_of_line_call(
    trace: ad.JVPTrace, primitive: OutOfLineCallPrimitive, f: lu.WrappedFun,
    tracers, params):
  assert primitive.multiple_results
  primals, tangents = util.unzip2(map(trace.to_primal_tangent_pair, tracers))
  which_nz = [     type(t) is not ad.Zero           for t in tangents]
  tangents = [t if type(t) is not ad.Zero else None for t in tangents]
  args, in_tree = tree_util.tree_flatten((primals, tangents))
  f_jvp = ad.jvp_subtrace(f, trace.tag)
  f_jvp, which_nz_out = ad.nonzero_tangent_outputs(f_jvp)
  f_jvp, out_tree = ad.traceable(f_jvp, in_tree)
  # update_params = call_param_updaters.get(call_primitive)
  # new_params = update_params(params, which_nz) if update_params else params

  key = params["key"]
  new_params = dict(params)
  new_params["key"] = dataclasses.replace(key,
      transformation_stack=key.transformation_stack + (JVPOf(tuple(which_nz)),))

  fun_and_args = (ad._update_annotation(f_jvp, f.in_type, which_nz),) + tuple(args)
  result = primitive.bind_with_trace(trace.parent_trace, fun_and_args, new_params)
  primal_out, tangent_out = tree_util.tree_unflatten(out_tree(), result)
  tangent_out = [ad.Zero.from_primal_value(p) if t is None else t
                  for p, t in zip(primal_out, tangent_out)]
  return [ad.maybe_jvp_tracer(trace, p, t) for p, t in zip(primal_out, tangent_out)]

ad.JVPTrace.process_out_of_line_call = jvp_process_out_of_line_call

@dataclasses.dataclass(frozen=True)
class JVPOf:
  nz: tuple[bool, ...]

  def __repr__(self):
    return f"jvp{self.nz}"
