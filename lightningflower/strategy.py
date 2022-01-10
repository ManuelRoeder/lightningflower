"""LightningFlower Strategy"""

import os
import flwr as fl
import numpy as np
import pytorch_lightning.utilities.argparse as arg_parser
from argparse import ArgumentParser, Namespace
from flwr.server.strategy import FedAvg
from lightningflower.utility import printf, load_params, boolean_string, create_dir
from lightningflower.config import LightningFlowerDefaults
from typing import Any, Union


class LightningFlowerBaseStrategy:
    """Base class for lightning flower strategies.

    Allows saving and loading of initial model weights,
    takes care of argument parsing.
    """

    def __init__(self, init_model, save_model, model_path):
        # create save/load directory
        create_dir(model_path)
        self.save_model = save_model
        self.init_model = init_model
        self.full_weights_path = os.path.join(model_path, LightningFlowerDefaults.WEIGHTS_FILENAME)
        if self.init_model:
            self.initial_parameters = load_params(self.full_weights_path)

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

    def save_weights(self, weights):
        """ Save the model weights if required and valid

        :param weights:
        :return:
        """
        if weights is not None and self.save_model:
            # Save aggregated_weights
            printf("Saving model weights")
            np.savez(self.full_weights_path, *weights)


class LightningFlowerFedAvgStrategy(LightningFlowerBaseStrategy, fl.server.strategy.FedAvg):
    @staticmethod
    def add_strategy_specific_args(parent_parser):
        # add base strategy related arguments
        parser = LightningFlowerBaseStrategy.add_strategy_specific_args(parent_parser)
        # FedAvg specific arguments
        parser.add_argument("--fraction_fit", type=float, default=0.1)
        parser.add_argument("--fraction_eval", type=float, default=0.1)
        parser.add_argument("--min_fit_clients", type=int, default=2)
        parser.add_argument("--min_eval_clients", type=int, default=2)
        parser.add_argument("--min_available_clients", type=int, default=2)
        parser.add_argument("--accept_failures", type=boolean_string, default=True)
        return parent_parser

    def __init__(self, fraction_fit,
                       fraction_eval,
                       min_fit_clients,
                       min_eval_clients,
                       min_available_clients,
                       accept_failures,
                       init_model,
                       save_model,
                       model_path):
        fl.server.strategy.FedAvg.__init__(self,
                                           fraction_fit=fraction_fit,
                                           fraction_eval=fraction_eval,
                                           min_fit_clients=min_fit_clients,
                                           min_eval_clients=min_eval_clients,
                                           min_available_clients=min_available_clients,
                                           accept_failures=accept_failures)

        LightningFlowerBaseStrategy.__init__(self,
                                             init_model=init_model,
                                             save_model=save_model,
                                             model_path=model_path)

    def aggregate_fit(self, rnd, results, failures):
        # dispatch weight aggregation from FedAvg strategy
        aggregated_weights = fl.server.strategy.FedAvg.aggregate_fit(self, rnd, results, failures)
        # save weights if required
        LightningFlowerBaseStrategy.save_weights(self, aggregated_weights)
        return aggregated_weights
