# What is this folder for?
When training QCNN, there might happen some unexpected break. Therefore, in
this folder is every trainingsprogress stored.

# How to open these files?
```python
import torch
file = "path to file location"
params: dict = torch.load(file, weights_only=False)
```

# What is stored in these files?
`params` is a dictionary with the following keys.
* `'number_of_qubits'`
* `'dataset'`
* `'dataset_name'`
* `'model_init_parameters'`
* `'parameters'`
* `'optimizer_momentum'`
* `'optimizer'`
* `'embedding_type'`
* `'cost_function'`
* `'loss_history'`
* `'epoch'`