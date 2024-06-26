import functools
from collections.abc import Sequence

from jax._src import ad_util
from jax._src import api_util
from jax._src import core
from jax._src import linear_util as lu
from jax._src import source_info_util
from jax._src.interpreters import ad
from jax._src.interpreters import partial_eval as pe
from jax._src.tree_util import tree_flatten, tree_unflatten
from jax._src.util import merge_lists, safe_map, safe_zip, split_list, partition_list

map = safe_map
zip = safe_zip


def noop(fun):
  @functools.wraps(fun)
  def wrapped(*args):
    args_flat, in_tree = tree_flatten(args)
    fun_flat, out_tree = api_util.flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
    avals_in = [core.raise_to_shaped(core.get_aval(x)) for x in args_flat]
    debug = pe.debug_info(fun, in_tree, out_tree, False, "noop")
    jaxpr, consts = _make_jaxpr(fun_flat, avals_in, debug)
    out_flat = noop_p.bind(*consts, *args_flat, jaxpr=jaxpr)
    return tree_unflatten(out_tree(), out_flat)

  return wrapped


def _make_jaxpr(
  fun: lu.WrappedFun,
  avals_in: Sequence[core.AbstractValue],
  debug_info: pe.DebugInfo | None = None,
):
  jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(fun, avals_in, debug_info)
  jaxpr = core.ClosedJaxpr(pe.convert_constvars_jaxpr(jaxpr), ())
  return jaxpr, consts


def noop_impl(*args, jaxpr: core.ClosedJaxpr):
  return core.jaxpr_as_fun(jaxpr)(*args)


def noop_abstract_eval(*args, jaxpr: core.ClosedJaxpr):
  del args
  return jaxpr.out_avals, jaxpr.effects


def noop_jvp(primals, tangents, jaxpr: core.ClosedJaxpr):
  nonzeros = [not isinstance(x, ad_util.Zero) for x in tangents]
  jvp_jaxpr, nonzeros_out = ad.jvp_jaxpr(jaxpr, nonzeros, False)



  tangents = [t for t in tangents if not isinstance(t, ad_util.Zero)]
  out_flat = noop_p.bind(*primals, *tangents, jaxpr=jvp_jaxpr)
  out_primals, out_tangents = split_list(out_flat, [len(nonzeros_out)])

  out_tangents_iter = iter(out_tangents)
  out_tangents = [
    next(out_tangents_iter) if nz else ad_util.Zero.from_value(x)
    for nz, x in zip(nonzeros_out, out_primals)
  ]

  return out_primals, out_tangents


def noop_partial_eval(trace, *tracers, jaxpr: core.ClosedJaxpr):
  in_unknowns = [not t.pval.is_known() for t in tracers]

  jaxpr_known, jaxpr_unknown, out_unknowns, res_avals = pe.partial_eval_jaxpr_nounits(
    jaxpr, in_unknowns, False
  )
  num_res = len(res_avals)

  in_known = [t.pval.get_known() for t in tracers if t.pval.is_known()]
  out_known = noop_p.bind(*in_known, jaxpr=jaxpr_known)
  out_known, res = split_list(out_known, [len(out_known) - num_res])

  in_tracers = [trace.instantiate_const(t) for uk, t in zip(in_unknowns, tracers) if uk]
  res_tracers = map(trace.new_instantiated_const, res)
  out_tracers = [
    pe.JaxprTracer(trace, pe.PartialVal.unknown(aval), None)
    for aval in jaxpr_unknown.out_avals
  ]
  params = dict(jaxpr=jaxpr_unknown)
  print(jaxpr_unknown)
  name_stack = source_info_util.current_name_stack()[len(trace.name_stack) :]
  source = source_info_util.current().replace(name_stack=name_stack)
  eqn = pe.new_eqn_recipe(
    res_tracers + in_tracers, out_tracers, noop_p, params, jaxpr_unknown.effects, source
  )
  for t in out_tracers:
    t.recipe = eqn

  return merge_lists(out_unknowns, out_known, out_tracers)


def noop_transpose(ct, *args, jaxpr: core.ClosedJaxpr):
  print(jaxpr)
  is_prim = [isinstance(x, ad.UndefinedPrimal) for x in args]
  res_avals, primal_avals = partition_list(is_prim, jaxpr.in_avals)
  primal_avals = map(core.raise_to_shaped, primal_avals)

  @lu.wrap_init
  def transposed(*args):
    res, cts_out = split_list(args, [len(res_avals)])
    primals = merge_lists(is_prim, res, [ad.UndefinedPrimal(a) for a in primal_avals])
    cts_in = ad.backward_pass(jaxpr.jaxpr, False, jaxpr.consts, primals, cts_out)
    cts_res, cts_in = partition_list(is_prim, cts_in)
    assert all(isinstance(x, ad_util.Zero) for x in cts_res)
    return map(ad.instantiate_zeros, cts_in)

  jaxpr_trans, consts = _make_jaxpr(transposed, res_avals + jaxpr.out_avals)
  cts_in = noop_p.bind(*consts, *ct, jaxpr=jaxpr_trans)

  return merge_lists(is_prim, [None] * len(res_avals), cts_in)


noop_p = core.Primitive("noop")
noop_p.multiple_results = True
noop_p.def_impl(noop_impl)
noop_p.def_effectful_abstract_eval(noop_abstract_eval)
ad.primitive_jvps[noop_p] = noop_jvp
ad.primitive_transposes[noop_p] = noop_transpose
pe.custom_partial_eval_rules[noop_p] = noop_partial_eval


if __name__ == "__main__":
  import jax
  import jax.numpy as jnp
  import numpy as np

  def fun0(x, y):
    return jnp.sum(np.array([[1.0, -0.5], [1.3, 5.3]], dtype=np.float32) @ x) * y

  fun = noop(fun0)

  # np.testing.assert_allclose(
  #   jax.jvp(fun0, (jnp.array([0.5, 0.3]),), (jnp.ones(2),)),
  #   jax.jvp(fun, (jnp.array([0.5, 0.3]),), (jnp.ones(2),)),
  # )

  jax.grad(fun, argnums=(0, 1))(jnp.array([0.5, 0.3]), 0.5)

  # print(jax.make_jaxpr(jax.grad(fun0, argnums=(0, 1)))(jnp.array([0.5, 0.3]), 0.5))
  # print(jax.make_jaxpr(jax.grad(fun, argnums=(0, 1)))(jnp.array([0.5, 0.3]), 0.5))
