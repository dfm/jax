
## linearize :: a, (a -> b) -> b, (T a -o T b)

import dataclasses
from collections.abc import Sequence
from typing import Callable

from jax._src import api_util
from jax._src import core
from jax._src import linear_util as lu
# from jax._src.ad_util import Zero
from jax._src.interpreters import ad
from jax._src.lax import lax
from jax._src.util import safe_map, safe_zip, unzip2
from jax._src.tree_util import tree_flatten, tree_unflatten

map = safe_map
zip = safe_zip


def linearize(fun: Callable, *primals):
  primals_flat, in_tree = tree_flatten(primals)
  fun = lu.wrap_init(fun)
  flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
  flat_fun = lin_impl(lin_subtrace(flat_fun))
  out_flat, lins = flat_fun.call_wrapped(
      primals_flat, [LinNode()] * len(primals_flat))
  print(out_flat)
  print(lins)
  return tree_unflatten(out_tree(), out_flat)

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
  in_tracers = [LinTracer(trace, x, LinNode(lin, None)) # if lin is not None else x
                for x, lin in zip(primals, lins)]
  ans = yield in_tracers, {}
  out_tracers = map(trace.full_raise, ans)
  yield unzip2((t.primal, t.lin) for t in out_tracers)

class LinTrace(core.Trace):
  def pure(self, val):
    return LinTracer(self, val, LinNode())

  lift = pure

  def sublift(self, val):
    return LinTracer(self, val.primal, val.lin)

  def process_primitive(self, primitive: core.Primitive, tracers, params):
    print(primitive)
    lin_rule = primitive_linearize_rules.get(primitive)
    if not lin_rule:
      msg = f"Linearization rule for '{primitive}' not implemented"
      raise NotImplementedError(msg)
    primals_in, lin_nodes_in = unzip2((t.primal, t.lin) for t in tracers)
    primals_out, lins_out = lin_rule(*primals_in, **params)
    if primitive.multiple_results:
      return [LinTracer(self, x, LinNode(lin, lin_nodes_in))
              for x, lin in zip(primals_out, lins_out)]
    else:
      return LinTracer(self, primals_out, LinNode(lins_out, lin_nodes_in))

@dataclasses.dataclass
class LinNode:
  lin: Callable | None = None
  input_lins: Sequence["LinNode"] | None = None

class LinTracer(core.Tracer):
  __slots__ = ["primal", "lin"]
  def __init__(self, trace, primal, lin):
    self._trace = trace
    self.primal = primal
    self.lin = lin

  @property
  def aval(self):
    return core.get_aval(self.primal)

  def full_lower(self):
    if self.lin is None:
      return core.full_lower(self.primal)
    else:
      return self

primitive_linearize_rules: dict[core.Primitive, Callable] = {}

def add_lin(x, y):
  return x + y, ad.add_tangents
primitive_linearize_rules[lax.add_p] = add_lin

def exp_lin(x):
  y = lax.exp_p.bind(x)
  return x, lambda t: y * t
primitive_linearize_rules[lax.exp_p] = exp_lin


if __name__ == "__main__":
  def f(x, y):
    return lax.exp_p.bind(lax.add_p.bind(x, y))

  def g(x, y, z):
    return f(f(x, y), z)

  lin = linearize(f, 0.1, 0.5)
  print(lin)
