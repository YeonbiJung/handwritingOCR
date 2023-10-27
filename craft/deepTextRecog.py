import torch
from models.base import ModelBase


class DTR(ModelBase):
    def __init__(self, cfg):
        super(DTR, self).__init__(cfg)
        self.cfg = cfg
        
    def forward(self, input):
        pass
        
    def training_step(self, batch, batch_nb):
        pass
    
    def validation_step(self, batch, batch_nb):
        pass
    
    def validation_epoch_end(self, outputs):
        pass
    
    def cal_loss(self, logits, targets):
        pass