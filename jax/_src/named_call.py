# Copyright 2025 The JAX Authors.
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

from collections.abc import Callable, Sequence
from functools import partial, wraps

from jax._src import api_util
from jax._src import core
from jax._src import linear_util as lu
from jax._src.interpreters import ad
from jax._src.interpreters import mlir


def named_call(
    name: str,
    fun: Callable,
    *,
    multiple_results: bool = False,
    static_argnums: Sequence[int] = (),
):
  static_argnums = set(static_argnums)

  def flat_fun(*args):
    out = fun(*args)
    return out if multiple_results else [out]

  @wraps(fun)
  def wrapped(*args):
    debug_info = api_util.debug_info(f"named_call {name}", fun, args, {})
    if static_argnums:
      dyn_argnums = [i for i in range(len(args)) if i not in static_argnums]
      f_, dyn_args = api_util.argnums_partial(
          lu.wrap_init(flat_fun, debug_info=debug_info), dyn_argnums, args,
          require_static_args_hashable=False)
    else:
      f_, dyn_args = lu.wrap_init(flat_fun, debug_info=debug_info), args
    out = named_call_p.bind(
        f_, *dyn_args, name=name, multiple_results=multiple_results)
    return out if multiple_results else out[0]

  return wrapped


named_call_p = core.CallPrimitive("named_call")
named_call_p.def_impl(core.call_impl)

def _named_call_pp_rule(eqn: core.JaxprEqn,
                        context: core.JaxprPpContext,
                        settings: core.JaxprPpSettings) -> core.pp.Doc:
  return core._pp_eqn(eqn, context, settings, params=["name", "call_jaxpr"])
core.pp_eqn_rules[named_call_p] = _named_call_pp_rule

# TODO(dfm): Add updaters to other interpreters. At least batching?
def drop_name_from_call(params, *_):
  new_params = dict(params)
  new_params.pop("name")
  new_params.pop("multiple_results")
  return new_params
ad.call_param_updaters[named_call_p] = drop_name_from_call
ad.call_linearize_param_updaters[named_call_p] = drop_name_from_call
ad.call_transpose_param_updaters[named_call_p] = drop_name_from_call

mlir.register_lowering(
    named_call_p, partial(mlir.core_call_lowering, name="named_call"))
