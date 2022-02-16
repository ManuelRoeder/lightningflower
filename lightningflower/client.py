"""LightningFlower Client"""
import flwr as fl
import pytorch_lightning as pl
import torch
import timeit

from collections import OrderedDict
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from lightningflower.config import LightningFlowerDefaults
from lightningflower.model import LightningFlowerModel
from torch.utils.data import DataLoader



class LightningFlowerClient(fl.client.Client):
    @staticmethod
    def add_client_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LightningFlowerClient")
        parser.add_argument("--host_address", type=str, default=LightningFlowerDefaults.HOST_ADDRESS)
        parser.add_argument("--max_msg_size", type=int, default=LightningFlowerDefaults.GRPC_MAX_MSG_LENGTH)
        parser.add_argument("--client_id", type=int, default=LightningFlowerDefaults.CLIENT_ID)
        parser.add_argument("--num_clients", type=int, default=LightningFlowerDefaults.NUM_CLIENTS)
        return parent_parser

    @staticmethod
    def update_state_dict(current_model_dict, parameters):
        params_dict = zip(current_model_dict.keys(), parameters)
        # in case of trouble with batch norm try torch.from_numpy(v)
        loaded_state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # allow mismatching shapes
        new_state_dict = {k: v if v.size() == current_model_dict[k].size() else current_model_dict[k] for k, v in
                          zip(current_model_dict.keys(), loaded_state_dict.values())}
        return new_state_dict

    def __init__(self,
                 model,
                 trainer_args,
                 c_id,
                 datamodule=None,
                 train_ds=None,
                 test_ds=None,
                 train_sampler=None):
        # make sure that the model is a lightningflowermodel
        assert isinstance(model, LightningFlowerModel)
        # check option datasets
        if train_ds is None:
            train_ds = []
        if test_ds is None:
            test_ds = []
        # client id
        self.c_id = c_id
        # assign local model
        self.localModel = model

        # datamodule or train/test sets
        if (train_ds is not None or test_ds is not None) and datamodule is not None:
            raise MisconfigurationException(
                "You cannot pass `train_dataloader` or `val_dataloaders` to `trainer.fit(datamodule=...)`"
            )

        self.datamodule = datamodule
        # optional training and test datasets
        self.train_ds = train_ds
        self.test_ds = test_ds
        # sampler used to draw examples
        self.train_sampler = train_sampler

        # trainer configuration
        self.trainer_config = trainer_args

    def get_weights(self):
        """Get model weights as a list of NumPy ndarrays."""
        ret_val = [val.cpu().numpy() for key, val in self.localModel.model.state_dict().items()]
        return ret_val

    def get_trainable_weights(self):
        """Get only trainable model weights as a list of NumPy ndarrays."""
        ret_val = [val.cpu().numpy() for key, val in self.localModel.model.state_dict().items()
                   if key not in self.localModel.fixed_model_parameters]
        return ret_val

    def get_parameters(self):
        """Return the current local model parameters.

        Returns
        -------
        ParametersRes
            The current local model parameters.
        """
        print(f"Client {self.c_id}: get_parameters")
        weights: fl.common.Weights = self.get_trainable_weights()
        parameters = fl.common.weights_to_parameters(weights)
        return fl.common.ParametersRes(parameters=parameters)

    @staticmethod
    def set_model_parameters(model, parameters):
        # we need to adapt for pot. new parameters and shape mismatching, may raise issues later
        # see https://github.com/pytorch/pytorch/issues/40859
        new_state_dict = LightningFlowerClient.update_state_dict(model.state_dict(), parameters)
        model.load_state_dict(new_state_dict, strict=False)

    def set_parameters(self, parameters):
        """Set the current local model parameters

        Parameters
        ----------
        parameters : model parameters

        """
        print(f"Client {self.c_id}: set_parameters")
        LightningFlowerClient.set_model_parameters(self.localModel.model, parameters)

    def fit(self, ins: FitIns) -> FitRes:
        """Refine the provided weights using the locally held dataset.

        Parameters
        ----------
        ins : FitIns
            The training instructions containing (global) model parameters
            received from the server and a dictionary of configuration values
            used to customize the local training process.

        Returns
        -------
        FitRes
            The training result containing updated parameters and other details
            such as the number of local training examples used for training.
        """
        print(f"Client {self.c_id}: fit")
        weights: fl.common.Weights = fl.common.parameters_to_weights(ins.parameters)
        # read configuration
        config = ins.config
        # begin time measurement
        fit_begin = timeit.default_timer()
        # update local client model parameters
        self.set_parameters(weights)

        data_loader = None
        if self.datamodule is None:
            # create dataloader on-the-fly
            shuffle_loading = self.train_sampler is None
            data_loader = DataLoader(dataset=self.train_ds,
                                     batch_size=self.trainer_config.batch_size_train,
                                     sampler=self.train_sampler,
                                     num_workers=self.trainer_config.num_workers,
                                     shuffle=shuffle_loading)

        else:
            data_loader = self.datamodule

        # training procedure
        trainer = pl.Trainer.from_argparse_args(self.trainer_config)
        trainer.fit(model=self.localModel.model, train_dataloaders=data_loader)
        # calculate nr. of examples used by Trainer for train
        num_train_examples = (data_loader.batch_size * trainer.num_training_batches)
        # return updated model parameters
        weights_prime: fl.common.Weights = self.get_trainable_weights()
        params_prime = fl.common.weights_to_parameters(weights_prime)
        metrics = {"duration": timeit.default_timer() - fit_begin}
        return FitRes(parameters=params_prime,
                      num_examples=num_train_examples,
                      metrics=metrics)

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the provided weights using the locally held dataset.

        Parameters
        ----------
        ins : EvaluateIns
            The evaluation instructions containing (global) model parameters
            received from the server and a dictionary of configuration values
            used to customize the local evaluation process.

        Returns
        -------
        EvaluateRes
            The evaluation result containing the loss on the local dataset and
            other details such as the number of local data examples used for
            evaluation.
        """
        print(f"Client {self.c_id}: evaluate")
        weights: fl.common.Weights = fl.common.parameters_to_weights(ins.parameters)
        # update local client model parameters
        self.set_parameters(weights)

        data_loader = None
        if self.datamodule is None:
            # create dataloader on-the-fly
            data_loader = DataLoader(dataset=self.test_ds,
                                     batch_size=self.trainer_config.batch_size_test,
                                     num_workers=self.trainer_config.num_workers,
                                     shuffle=False)
        else:
            data_loader = self.datamodule
        # evaluation procedure
        trainer = pl.Trainer.from_argparse_args(self.trainer_config)
        test_result = trainer.test(model=self.localModel.model, dataloaders=data_loader)
        # obtain result of first train_loader
        train_loader_0_result = test_result[0]
        test_loss = train_loader_0_result["test_loss"]
        accuracy = train_loader_0_result["test_acc"]
        # calculate nr. of examples used by Trainer for test
        num_test_examples = data_loader.batch_size * trainer.num_test_batches[0]
        metrics = {"accuracy": accuracy}
        return EvaluateRes(loss=test_loss,
                           num_examples=num_test_examples,
                           metrics=metrics)
