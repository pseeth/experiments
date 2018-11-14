# Architecture

## Modules

- `RecurrentStack`
- `ConvolutionalStack`
- `MelProjection`
- `Embedding`
- `Clusterer`
- `Centroid`

## Train

### Trainer

#### Parameters

- `target_directory` (`str`) - directory to store logs, checkpoints
- `input_keys` (`str[]`) - keys specifying dataset values used as input to network
- `output_keys` (`str[]`) - keys specifying dataset values used for validation/trianing against computation
- `model` (`torch.nn.modules.module`) -
- `dataset` (`torch.utils.data.Dataset`) - dictionary of `<str, any>`
- `val_dataset` (`torch.utils.data.Dataset`) -
- `loss_function` (`enum[]`) - `LossFunction` enum, possible values - [`L1`, `MSE`, `KL`, `DPCL`]
- `optimizer` (`enum`) - `Optimizer`, possible values - [`Adam`, `RMSProp`, `SGD`]
- `options` (`<str, any>`) - dictionary of possible options
  - `num_epochs` (`int`)
  - `batch_size` (`int`)

