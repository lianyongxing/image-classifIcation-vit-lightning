import torch.nn as nn
from transformers import ViTModel, ViTConfig, ViTImageProcessor
import warnings
import pytorch_lightning as pl

warnings.filterwarnings('ignore')


class VitClsModel(nn.Module):

    def __init__(self, path, classes=2):
        super(VitClsModel, self).__init__()
        self.config = ViTConfig.from_pretrained(path)  # 导入模型超参数
        self.vit = ViTModel.from_pretrained(path) 
        self.fc = nn.Linear(self.config.hidden_size, classes)  
        self.processor = ViTImageProcessor.from_pretrained(path)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values)
        out_pool = outputs[0]  
        logit = self.fc(out_pool[:, 0, :]) 
        return logit