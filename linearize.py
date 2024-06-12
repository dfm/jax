
## linearize :: a, (a -> b) -> b, (T a -o T b)

from typing import Callable

from jax._src import api_util
from jax._src import core
from jax._src import linear_util as lu
from jax._src.ad_util import Zero
from jax._src.lax import lax
from jax._src.util import safe_map, safe_zip, unzip2
from jax._src.tree_util import tree_flatten
from jax._src.interpreters import partial_eval as pe

map = safe_map
zip = safe_zip


def linearize(fun: Callable, *primals):
  primals_flat, in_tree = tree_flatten(primals)
  fun = lu.wrap_init(fun)
  flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
  flat_fun = lin_impl(lin_subtrace(flat_fun))
  return flat_fun.call_wrapped(primals_flat, [0.0] * len(primals_flat))

@lu.transformation
def lin_impl(primals, lins):
  with core.new_main(LinTrace) as main:
    out_primals, lin_func = yield (main, primals, lins), {}
    del main
  yield out_primals, lin_func

@lu.transformation
def lin_subtrace(main, primals, lins):
  trace = LinTrace(main, core.cur_sublevel())
  for x in primals:
    if isinstance(x, core.Tracer):
      assert x._trace.level < trace.level, "todo: error"
  in_tracers = [LinTracer(trace, x, lin) if lin is not None else x
                for x, lin in zip(primals, lins)]
  ans = yield in_tracers, {}
  out_tracers = map(trace.full_raise, ans)
  yield unzip2((t.primal, t.linear_jaxpr) for t in out_tracers)


class LinTrace(core.Trace):
  def pure(self, val):
    return LinTracer(self, val, None)

  lift = pure

  def sublift(self, val):
    return LinTracer(self, val.primal, val.linear_jaxpr)

  def process_primitive(self, primitive: core.Primitive, tracers, params):
    lin_rule = primitive_linearize_rules.get(primitive)
    if not lin_rule:
      msg = f"Linearization rule for '{primitive}' not implemented"
      raise NotImplementedError(msg)
    primals_in, lin_jaxpr_in = unzip2((t.primal, t.linear_jaxpr) for t in tracers)
    primals_out, lins_out = lin_rule(*primals_in, **params)
    pvals_in = tuple(pe.PartialVal.unknown(core.get_aval(x).at_least_vspace())
                     for x in primals_in)
    jaxpr_out = pe.trace_to_jaxpr_nounits(lu.wrap_init(lins_out), pvals_in)
    print(jaxpr_out)
    assert 0

    if primitive.multiple_results:
      return [LinTracer(self, x, lin) for x, lin in zip(primals_out, lins_out)]
    else:
      return LinTracer(self, primals_out, lins_out)


primitive_linearize_rules: dict[core.Primitive, Callable] = {}

class LinTracer(core.Tracer):
  __slots__ = ["primal", "linear_jaxpr"]
  def __init__(self, trace, primal, linear_jaxpr):
    self._trace = trace
    self.primal = primal
    self.linear_jaxpr = linear_jaxpr

  @property
  def aval(self):
    return core.get_aval(self.primal)

  def full_lower(self):
    if self.linear_jaxpr is None:
      return core.full_lower(self.primal)
    else:
      return self

def add_tangents(args):
  xt, yt = args
  if isinstance(xt, Zero):
    return yt
  elif isinstance(yt, Zero):
    return xt
  else:
    return xt + yt

def add_lin(*primals):
  return lax.add_p.bind(*primals), add_tangents

primitive_linearize_rules[lax.add_p] = add_lin


if __name__ == "__main__":
  def f(args):
    x, y = args
    return lax.add_p.bind(x, y)

  lin = linearize(f, (0.1, 0.5))
  print(lin)
