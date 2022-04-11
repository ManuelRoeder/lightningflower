"""LightningFlower Model"""
import flwr as fl
import pytorch_lightning as pl


class LightningFlowerModel:
    def __init__(self, model, name="", strict_params=False):
        # Check if the model is a Lightning Module
        assert isinstance(model, pl.LightningModule)
        # Internally persist the model
        self.model = model
        # Model name used for saving weights
        self.name = name
        # model parameter that are not sent to the server, e.g. frozen weights
        self.fixed_model_parameters = self.__get_fixed_model_params()
        # define if only strict model parameters are allowed for client-server transmissions
        # number of server model parameters need to be the same as number of client model parameters
        self.strict = strict_params

    def __get_fixed_model_params(self):
        param_list = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                param_list.append(name)
        return param_list

    def get_flwr_params(self):
        # get the weights
        weights = [val.cpu().numpy() for key, val in self.model.state_dict().items()]
        return fl.common.weights_to_parameters(weights)
