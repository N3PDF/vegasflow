---
name: Bug report
about: Create a report to help us improve
title: ''
labels: bug
assignees: ''

---

### Description

Please, describe briefly what the issue is

### Code example

If possible, write a minimum working example that reproduces the bug,
e.g:

```python
import vegasflow
vegasflow.broken_function()
```

### Additional information

Does the problem occur in CPU or GPU?
If GPU, how many? Which version of Cuda do you have?

```bash
nvcc --version
```

Please include the version of vegasflow and tensorflow that you are running. Running the following python script will produce useful information:

```python
import tensorflow as tf
import vegasflow

print(f"Vegasflow: {vegasflow.__version__}")
print(f"Tensorflow: {tf.__version__}")
print(f"tf-mkl: {tf.python.framework.test_util.IsMklEnabled()}")
print(f"tf-cuda: {tf.test.is_built_with_cuda()}")
print(f"GPU available: {tf.test.is_gpu_available()}")
```
