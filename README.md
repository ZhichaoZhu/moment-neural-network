# Moment Neural Network

Moment Neural Network (MNN) is a PyTorch-based framework for building,
training, and simulating neural networks that propagate second-order
statistics. Instead of tracking only mean neural activity, an MNN tracks
both mean activity and covariance, making it suitable for studying
correlated neural variability in spiking neural systems.

The project is motivated by stochastic neural computing for spiking
neural networks (SNNs). Once trained, MNN parameters can be used to
recover a corresponding SNN without an additional fine-tuning phase. The
trained model is intended to capture realistic firing statistics,
including firing-rate distributions, Fano factors, and weak pairwise
correlations.

For full details, see the associated publication:
https://doi.org/10.1093/pnasnexus/pgaf284

## What This Project Provides

- Differentiable MNN layers that transform both mean and covariance.
- Moment activation functions derived from leaky integrate-and-fire
  neuron statistics.
- MLP model wrappers for MNN, SNN-like, ANN, and mean-only baselines.
- YAML-driven training utilities compatible with standard PyTorch
  optimizers and TorchVision datasets.
- Utilities to recover and simulate SNNs from trained MNN checkpoints.
- Tutorials and publication-specific experiment scripts.

## Installation

Install the project through <code>pip</code>:

```bash
pip install moment-neural-network
```


Alternatively, clone the repository and install it from the repository root:

```bash
git clone git@github.com:BrainsoupFactory/moment-neural-network.git
cd moment-neural-network
python -m pip install -e .
```

The package metadata requires Python 3.8 or newer and installs:

- `torch`
- `torchvision`
- `numpy`
- `scipy`
- `PyYAML`
- `tqdm`

Optional extras:

```bash
python -m pip install -e ".[analysis]"
python -m pip install -e ".[loihi]"
```

The original development environment used PyTorch 1.12.1, TorchVision
0.13.1, SciPy 1.7.3, PyYAML 6.0, and NumPy 1.22.3. Newer versions may
work, but use those versions as a starting point for
reproducibility-sensitive experiments.

CUDA is optional but recommended for larger MNNs and SNN simulations with
many trials.

## Repository Structure

- `mnn/mnn_core`: core MNN math and PyTorch autograd bindings.
- `mnn/mnn_core/nn`: layers, activations, losses, batch normalization,
  pooling, and functional utilities.
- `mnn/models`: model wrappers, including `MnnMlp`, `SnnMlp`, `AnnMlp`,
  and mean-only variants.
- `mnn/snn`: MNN-to-SNN conversion and SNN simulation utilities.
- `mnn/snn/base`: LIF neurons, current generators, monitors, and probes.
- `mnn/utils/training_tools`: config loading, dataloaders, training,
  checkpointing, logging, and resume helpers.
- `mnn/utils/dataloaders`: dataset loaders for MNIST and related
  examples.
- `mnn/analysis`: analysis and visualization helpers.
- `docs`: Sphinx documentation and tutorial notebooks.
- `examples/mnist`: runnable MNIST training and SNN validation example.
- `publications`: scripts and configs used for paper-specific
  experiments.

## Quick Start: MNIST

Create output and data directories:

```bash
mkdir -p checkpoint data
```

Run the MNIST example:

```bash
python examples/mnist/mnist.py --config=examples/mnist/mnist_config.yaml
```

Important: the current `examples/mnist/mnist.py` defaults to running SNN
simulation in `main()`, while `train_mnist(config)` is commented out.
For a first training run, edit `main()` to train:

```python
def main():
    config = utils.training_tools.deploy_config()
    train_mnist(config)
    # mnn2snn_simulation(config)
```

After training, switch back to SNN validation:

```python
def main():
    config = utils.training_tools.deploy_config()
    # train_mnist(config)
    mnn2snn_simulation(config)
```

By default, outputs are written under:

```text
checkpoint/mnist/
```

Typical files include:

- `mnn_net_config.yaml`: saved copy of the effective config.
- `mnn_net_log.txt`: training and validation log.
- `mnn_net.pth`: latest checkpoint.
- `mnn_net_best_model.pth`: best validation checkpoint.
- `mnn_net_snn_validate_result/`: SNN simulation outputs.

## Configuration Workflow

The YAML config controls model construction, optimization, data
transforms, input encoding, and output paths.

Important sections:

- `MODEL`: model family and architecture. The MNIST example uses
  `arch: mnn_mlp` and `mlp_type: mnn_mlp`.
- `CRITERION`: loss function. The example uses `CrossEntropyOnMean`.
- `OPTIMIZER`: PyTorch optimizer name and arguments. The example uses
  `AdamW`.
- `DATAAUG_TRAIN` and `DATAAUG_VAL`: TorchVision transform pipelines.
- `input_prepare`: input-to-moment encoding. `flatten_poisson` flattens
  images and creates diagonal covariance from Poisson-like rates.
- `scale_factor`: multiplier applied to input firing-rate-like values.
- `background_noise`: optional diagonal covariance offset.
- `dump_path`, `dir`, and `save_name`: output checkpoint location and
  file prefix.

Command-line arguments are parsed by
`mnn/utils/training_tools/general_prepare.py`. Example:

```bash
python examples/mnist/mnist.py \
  --config=examples/mnist/mnist_config.yaml \
  --bs=128 \
  --epochs=10 \
  --cpu
```

Values loaded from the YAML config can overwrite command-line defaults
when the same key is present in the config.

## Using MNN Layers Directly

For custom research code, you can bypass the wrapper pipeline and use
MNN layers directly:

```python
import torch
from mnn.mnn_core.nn.linear import LinearDuo
from mnn.mnn_core.nn.custom_batch_norm import CustomBatchNorm1D
from mnn.mnn_core.nn.activation import OriginMnnActivation
from mnn.mnn_core.nn.criterion import CrossEntropyOnMean

class SimpleMNN(torch.nn.Module):
    def __init__(self, input_size=784, hidden_size=100, output_size=10):
        super().__init__()
        self.linear = LinearDuo(input_size, hidden_size)
        self.bn = CustomBatchNorm1D(hidden_size)
        self.act = OriginMnnActivation()
        self.readout = LinearDuo(hidden_size, output_size, bias=True)

    def forward(self, inputs):
        u, cov = inputs
        u, cov = self.linear(u, cov)
        u, cov = self.bn(u, cov)
        u, cov = self.act(u, cov)
        return self.readout(u, cov)

def encode_poisson_images(images, scale=1.0):
    mean = torch.flatten(images, start_dim=1) * scale
    cov = torch.diag_embed(torch.abs(mean))
    return mean, cov

model = SimpleMNN()
criterion = CrossEntropyOnMean()
```

See `docs/tutorials/tutorial_training_mnn_vanilla.ipynb` for a more
complete direct-use example.

## SNN Reconstruction and Simulation

After a model has been trained and checkpointed, use
`snn.functional.MnnSnnValidate` or a task-specific subclass such as the
MNIST example's `MnistSnnValidate`.

Typical simulation parameters:

- `running_time`: simulated duration in ms.
- `dt`: simulation time step.
- `num_trials`: number of SNN trials simulated in parallel.
- `input_type`: `poisson` or `gaussian`.
- `resume_best`: whether to load the best or latest checkpoint.
- `pregenerate`: whether to pregenerate stochastic inputs.

Simulation outputs are saved as:

- `.snnval`: summary data such as MNN output, SNN output, spike counts,
  target, prediction, run time, and input type.
- `.spt`: recorded spike trains as sparse tensors.

## Tutorials

The Sphinx documentation and tutorials are in `docs/`.

Recommended reading order:

1. `docs/tutorials/tutorial_moment_activation.ipynb`
2. `docs/tutorials/tutorial_training_mnn.ipynb`
3. `docs/tutorials/tutorial_training_mnn_vanilla.ipynb`
4. `docs/tutorials/tutorial_EI_network.ipynb`

## Publication Code

The `publications` folder contains experiment-specific code:

- `Qi2024pre`: code for *Moment neural network and an efficient numerical
  method for modeling irregular spiking activity*.
- `Zhu2024plos`: training code for *Learning to integrate parts for whole
  through correlated neural variability*.
- `Zhu2024neco`: training code for *Toward a Free-Response Paradigm of
  Decision Making in Spiking Neural Networks*.
- `Ma2023neco`: additional publication material.

The reusable project API lives under `mnn`; publication folders are most
useful for reproducing paper-specific experiments.

## Current Limitations

- The most mature model path is MLP-based MNN training.
- Full covariance scales quadratically with layer width, so memory use
  can become large.
- The convolution implementation is present but less developed than the
  MLP stack.
- Some publication scripts are environment-specific and may need path,
  dataset, or device edits before running.
- SNN simulation is slower than an MNN forward pass because it advances a
  stochastic process over many time steps.

## Lead Authors

- **Zhichao Zhu** - *Chief Architect* -
  [Zhichao Zhu](https://github.com/ZhichaoZhu)
- **Yang Qi** - *Lead Algorithm Design* -
  [Yang Qi](https://github.com/qiyangku)

## License

This project is licensed under the Apache License 2.0. See
[LICENSE](LICENSE) for details.
