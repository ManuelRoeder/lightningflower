"""LightningFlower Server"""
import flwr as fl
from flwr.server.client_manager import SimpleClientManager
from lightningflower.config import LightningFlowerDefaults
from lightningflower.strategy import LightningFlowerFedAvgStrategy


class LightningFlowerServer(fl.server.Server):
    @staticmethod
    def add_server_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LightningFlowerServer")
        parser.add_argument("--host_address", type=str, default=LightningFlowerDefaults.HOST_ADDRESS)
        parser.add_argument("--num_rounds", type=int, default=LightningFlowerDefaults.NR_ROUNDS)
        parser.add_argument("--max_msg_size", type=int, default=LightningFlowerDefaults.GRPC_MAX_MSG_LENGTH)
        return parent_parser

    def __init__(self, strategy=None):
        # create the client manager
        client_manager = SimpleClientManager()
        # apply default strategy
        if strategy is None:
            strategy = LightningFlowerFedAvgStrategy()
        super().__init__(client_manager=client_manager, strategy=strategy)

