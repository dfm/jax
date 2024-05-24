(profiling-and-performance)=
# Profiling and performance

```{note}
This is a placeholder for a section in the new {ref}`jax-tutorials-draft`.

For the time being, you may find some related content in the old documentation:
- {doc}`../profiling`
- {doc}`../device_memory_profiling`
- {doc}`../transfer_guard`
```

## Outline

1. Benchmarking
2. Inspecting lowered and compiled HLO
3. Profiling
    - Perfetto
    - Tensorboard
    - Nsight

## Notes

- Use a simple MLP as our example to make it familiar to users and slightly non-trivial
- Will perhaps dupe the FAQ discussion


One of our favorite things about JAX is that it typically has excellent runtime performance.

To demonstrate the all the tools that we will discuss here, let's start by defining a simple [multilayer perceptron (MLP)](https://en.wikipedia.org/wiki/Multilayer_perceptron), which might be familiar to some users, and just non-trivial enough to be a useful demonstration.

```python
import numpy as np
import jax
import jax.numpy as jnp

# Define a very simple multilayer perceptron (MLP) model in pure JAX
def mlp(layers, x):
  for w in layers[:-1]:
    x = jax.nn.relu(x @ w)
  return x @ layers[-1]

# Define some inputs using numpy so that they live on the CPU. Note that we
# use the float32 data type here because that's JAX's default precision
n = 128
batch_np = np.ones((1024, n), dtype=np.float32)
layers_np = []
for dim in [4, 8, 12]:
  layers_np.append(np.ones((n, dim), dtype=np.float32))
  n = dim
```

## Microbenchmarks

See also {ref}`faq-benchmark`.

### Data transfer

The first thing to remember when benchmarking JAX code is that it takes time to transfer data from the CPU to an accelerator and back again.
To isolate this operation, you can use the {func}`jax.device_put` function, to explicitly place the data on your accelerator.
Since this tutorial is automatically executed on a machine without any hardware accelerator, the timings in the following cell are not relevant, but TODO(dfm) add numbers for colab here.

```python
%time layers_jax, batch_jax = jax.device_put((layers_np, batch_np))
```

### Just-in-time compilation

The next 

```python
%timeit jax.jit(mlp).lower(layers_jax, batch_jax).compile()
```

```python
%timeit jax.block_until_ready(jax.jit(mlp)(layers_jax, batch_jax))
```

### Asynchronous dispatch

Don't forget to warm up:

```python
compiled_mlp = jax.jit(mlp)
_ = jax.block_until_ready(compiled_mlp(layers_jax, batch_jax))
```

And block:

```python
%timeit jax.block_until_ready(compiled_mlp(layers_jax, batch_jax))
```

```python

```

```python

```

```python

```

## Profiling with Perfetto

```python
with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
  jax.block_until_ready(compiled_mlp(layers_jax, batch_jax))
```

```python

```

```python

```

```python

```

## TensorBoard

```python
with jax.profiler.trace("/tmp/tensorboard"):
  for n in range(500):
    jax.block_until_ready(compiled_mlp(layers_jax, batch_jax))
```

```python

```
