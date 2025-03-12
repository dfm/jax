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

import contextlib
from typing import Any

import numpy as np

from jax import lax
from jax._src import core
from jax._src.pjit import pjit_p
from jax._src.named_call import named_call_p


@contextlib.contextmanager
def eval_using_numpy(allow_xla: bool = False):
  with core.set_current_trace(NumpyTrace(allow_xla=allow_xla)):
    yield


class NumpyTrace(core.Trace):
  def __init__(self, allow_xla: bool = False):
    self.allow_xla = allow_xla

  def process_primitive(self, primitive, tracers, params):
    print(f"Interpreting {primitive} using numpy")
    rule = primitive_numpy_rules.get(primitive)
    if not rule:
      if self.allow_xla:
        with core.set_current_trace(core.eval_trace):
          return primitive.impl(*tracers, **params)
      else:
        raise NotImplementedError(
            f"Numpy evaluation rule for {primitive} not implemented")
    with core.set_current_trace(self):
      return rule(*tracers, **params)

  def process_call(self, primitive, f, tracers, params):
    if primitive == named_call_p:
      name = params["name"]
      multiple_results = params["multiple_results"]
      print(f"Interpreting named call {name} using numpy")
      rule = named_call_numpy_rules.get(name)
      if not rule:
        raise NotImplementedError(
            f"Numpy evaluation rule for named call {name} not implemented")
      out = rule(*tracers)
      return out if multiple_results else [out]
    with core.set_current_trace(self):
      return f.call_wrapped(*tracers)

  def process_map(self, primitive, f, tracers, **_):
    del primitive  # unused
    with core.set_current_trace(self):
      return f.call_wrapped(*tracers)

  def process_custom_transpose(self, primitive, call, tracers, **_):
    del primitive  # unused
    with core.set_current_trace(self):
      return call.call_wrapped(*tracers)

  def process_custom_jvp_call(self, primitive, fun, jvp, tracers, **_):
    del primitive, jvp  # unused
    with core.set_current_trace(self):
      return fun.call_wrapped(*tracers)

  def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers, **_):  # pytype: disable=signature-mismatch
    del primitive, fwd, bwd  # unused
    with core.set_current_trace(self):
      return fun.call_wrapped(*tracers)


primitive_numpy_rules: dict[core.Primitive, Any] = {}

def pjit_numpy_rule(*args, jaxpr, **_):
  # with core.set_current_trace(numpy_trace):
  return core.jaxpr_as_fun(jaxpr)(*args)
primitive_numpy_rules[pjit_p] = pjit_numpy_rule

primitive_numpy_rules[lax.add_p] = np.add
primitive_numpy_rules[lax.sub_p] = np.subtract
primitive_numpy_rules[lax.mul_p] = np.multiply
primitive_numpy_rules[lax.div_p] = np.divide
primitive_numpy_rules[lax.sin_p] = np.sin
primitive_numpy_rules[lax.cos_p] = np.cos

def convert_element_type_numpy_rule(x, *, new_dtype, **_):
  return np.asarray(x, dtype=new_dtype)
primitive_numpy_rules[lax.convert_element_type_p] = convert_element_type_numpy_rule

named_call_numpy_rules = {}
named_call_numpy_rules["jax.numpy.searchsorted"] = np.searchsorted
