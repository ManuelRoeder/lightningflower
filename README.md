![Alt text](lightningflower_logo.PNG?raw=true "Logo")

# LightingFlower


Pre-packaged federated learning framework using Flower and
PyTorch-Lightning.

## Installation

To install this library, simply run the following command:

```sh
pip install lightningflower
```

**Installing the lightningflower framework should automatically install suitable versions
of** [Flower] **and** [PyTorch-Lightning].


##Features

### Integrated Argument Parser
LightningFlower provides integrated argument parsing for data, server and client handling:

```
# initialize the argument parser
parser = ArgumentParser()

# Data-specific arguments like batch size
parser = LightningFlowerData.add_data_specific_args(parser)

# Trainer-specific arguments like number of epochs
parser = pl.Trainer.add_argparse_args(parser)

# Client-specific arguments like host address
parser = LightningFlowerClient.add_client_specific_args(parser)

# Parse arguments
args = parser.parse_args()
```

### LightningFlowerBaseStrategy
A basic strategy to enable saving and loading of model weights as well as resuming the training procedure.

### Full Pytorch-Lightning Trainer integration
LightningFlower supports full use of the Pytorch-Lighting training routines on federated Flower clients:

```
# Configure the client trainer
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

# Define a Pytorch-Lighting compatible model to train
model = pl.RandomModel()

# Create a client, pass Trainer configuration to and model to client
LightingFlowerClient(model=model, trainer_args=args, ...)
```

### Federated transfer learning / backbone support
LightningFlower enables transfer learning by only transmitting trainable model parameters
from server to clients, saving bandwidth and computation time.


## ToDo
This is a work in progress.