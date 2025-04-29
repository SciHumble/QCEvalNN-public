# What is this folder for?
The `Evaluation` class stores here the results of classical and quantum.


# How to open these files?
```python
import pandas as pd
file = "path to file location"
data: pd.DataFrame = pd.read_csv(file)
```

# What is stored in these files?
The columns of the `DataFrame` `data` are different for classical or quantum.

## Classical
* `'model_name'`
* `'learning_rate'`
* `'batch_size'`
* `'epochs'`
* `'accurcay'`
* `'training_time'`
* `'flops'` [differs]
* `'parameters'`
* `'input_size'`
* `'layers'`
* `'date'`
* `'optimizer'`
* `'criterion'`
* `'dataset_name'`
* `'feature_reduction'`
* `'loss_history'`
* `'autosafe_file'`

## Quantum
* `'model_name'`
* `'learning_rate'`
* `'batch_size'`
* `'epochs'`
* `'accurcay'`
* `'training_time'`
* `'gates'` [differs]
* `'parameters'`
* `'input_size'`
* `'layers'`
* `'date'`
* `'optimizer'`
* `'criterion'`
* `'dataset_name'`
* `'compacted_dataset'` [only quantum]
* `'feature_reduction'`
* `'single_gate_error_rate'` [only quantum]
* `'cnot_error_rate` [only quantum]
* `'single_check_accuracy'` [only quantum]
* `'triple_check_accuracy'` [only quantum]
* `'quintil_check_accuracy'` [only quantum]
* `'predictions'` [only quantum]
* `'labels'` [only quantum]
* `'convolutional_layer'` [only quantum]
* `'pooling_layer'` [only quantum]
* `'loss_history'`
* `'autosafe_file'`
* `'embedding'` [only quantum]

