"""LightningFlower Model"""
import pytorch_lightning as pl


class LightningFlowerModel:
    def __init__(self, model, name=""):
        # Check if the model is a Lightning Module
        assert isinstance(model, pl.LightningModule)
        # Internally persist the model
        self.model = model
        # Model name used for saving weights
        self.name = name
        # model parameter that are not sent to the server, e.g. frozen weights
        self.fixed_model_parameters = self.__get_fixed_model_params()

    def __get_fixed_model_params(self):
        param_list = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                param_list.append(name)
        return param_list

