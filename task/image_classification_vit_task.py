# -*- coding: utf-8 -*-
# @Time    : 4/6/23 8:39 PM
# @Author  : LIANYONGXING
# @FileName: ViTImageClassificationModel.py

import pytorch_lightning as pl
from transformers import AdamW
import torch.nn as nn
from model.vit import VitClsModel
from dataset.basic_datasets import build_dataloader


class ViTImageClassificationModel(pl.LightningModule):
    def __init__(self, args):
        super(ViTImageClassificationModel, self).__init__()
        self.args = args
        self.vit = VitClsModel(self.args.pretrain_path)
        self.get_dataloader()
        # self.vit = VitClsModel("/Users/user/Desktop/model_file/vit-base-patch16-224")

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct / pixel_values.shape[0]

        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)

        return loss

    def configure_optimizers(self):
        # We could make the optimizer more fancy by adding a scheduler and specifying which parameters do
        # not require weight_decay but just using AdamW out-of-the-box works fine
        return AdamW(self.parameters(), lr=5e-5)

    def get_dataloader(self):
        self.train_dl, self.valid_dl = build_dataloader('/Users/user/Desktop/model_ViT/testdats.csv', batch_size=self.args.batch_size)

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.valid_dl

