"""LightningFlower Strategy"""
import flwr as fl
from flwr.server.strategy import FedAvg
import numpy as np
import os
import pytorch_lightning.utilities.argparse as arg_parser
from argparse import ArgumentParser, Namespace
from lightningflower.utility import printf, load_params, boolean_string, create_dir
from lightningflower.config import LightningFlowerDefaults
from typing import Any, Union, Dict


class LightningFlowerBaseStrategy:
    """Base class for lightning flower strategies.

    Allows saving and loading of initial model weights,
    takes care of argument parsing.
    """

    def __init__(self, init_model, save_model, model_path, model_name, dataset_name):
        # create save/load directory
        create_dir(model_path)
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.save_model = save_model
        self.init_model = init_model
        self.full_weights_path = os.path.join(model_path,
                                              self.model_name +
                                              "_" +
                                              self.dataset_name +
                                              LightningFlowerDefaults.WEIGHTS_FILE_ENDING)
        if self.init_model:
            loaded_params = load_params(self.full_weights_path)
            if loaded_params is not None:
                self.initial_parameters = loaded_params

    @classmethod
    def from_argparse_args(cls: Any, args: Union[Namespace, ArgumentParser], **kwargs) -> Any:
        """ Creates a LightingFlowerStrategy object from argument parser

        :param args: Arguments
        :param kwargs: Keyword Arguments
        :return: Configured LightningFlowerStrategy object
        """
        return arg_parser.from_argparse_args(cls, args, **kwargs)

    @staticmethod
    def add_strategy_specific_args(parent_parser):
        """ Add dedicated argument group for strategy implementation

        Example:
            parser = parent_parser.add_argument_group("MyStrategy")

            parser.add_argument('--argument', default=0, type=int, help="This is an argument")

            return parser

        :param parent_parser: The parent parser to add argument group
        :return: The updated parent parser
        """
        # LightningFlowerBaseStrategy specific arguments
        parser = parent_parser.add_argument_group("LightningFlowerBaseStrategy")
        parser.add_argument('--init_model', default=False, type=boolean_string,
                            help="Load pretrained global weights if True")
        parser.add_argument("--save_model", default=False, type=boolean_string,
                            help="Save global model weights after each evaluation round")
        parser.add_argument("--model_path", default=LightningFlowerDefaults.WEIGHTS_FOLDER, type=str,
                            help="Path to load/save weights to")
        return parent_parser

    @staticmethod
    def fit_round(rnd: int) -> Dict:
        """Send round number to client."""
        print("Starting new round " + str(rnd))
        return {"rnd": rnd}

    def save_weights(self, weights):
        """ Save the model weights if required and valid

        :param weights:
        :return:
        """
        if weights is not None and self.save_model:
            # Save aggregated_weights
            printf("Saving weights for model " + self.model_name + " using " + self.dataset_name)
            np.savez(self.full_weights_path, *weights)


class LightningFlowerFedAvgStrategy(LightningFlowerBaseStrategy, FedAvg):
    @staticmethod
    def add_strategy_specific_args(parent_parser):
        # add base strategy related arguments
        parser = LightningFlowerBaseStrategy.add_strategy_specific_args(parent_parser)
        # FedAvg specific arguments
        parser.add_argument("--fraction_fit", type=float, default=0.5)
        parser.add_argument("--fraction_eval", type=float, default=0.5)
        parser.add_argument("--min_fit_clients", type=int, default=2)
        parser.add_argument("--min_eval_clients", type=int, default=2)
        parser.add_argument("--min_available_clients", type=int, default=2)
        parser.add_argument("--accept_failures", type=boolean_string, default=True)
        return parent_parser

    def __init__(self,
                 fraction_fit,
                 fraction_eval,
                 min_fit_clients,
                 min_eval_clients,
                 min_available_clients,
                 accept_failures,
                 init_model,
                 save_model,
                 model_path,
                 model_name,
                 dataset_name):
        FedAvg.__init__(self,
                        fraction_fit=fraction_fit,
                        fraction_eval=fraction_eval,
                        min_fit_clients=min_fit_clients,
                        min_eval_clients=min_eval_clients,
                        min_available_clients=min_available_clients,
                        accept_failures=accept_failures)

        LightningFlowerBaseStrategy.__init__(self,
                                             init_model=init_model,
                                             save_model=save_model,
                                             model_path=model_path,
                                             model_name=model_name,
                                             dataset_name=dataset_name)

        if self.on_fit_config_fn is None:
            self.on_fit_config_fn = LightningFlowerBaseStrategy.fit_round

    def aggregate_fit(self, rnd, results, failures):
        # dispatch weight aggregation from FedAvg strategy
        aggregated_weights = FedAvg.aggregate_fit(self, rnd, results, failures)
        # save weights if required
        LightningFlowerBaseStrategy.save_weights(self, aggregated_weights)
        return aggregated_weights
