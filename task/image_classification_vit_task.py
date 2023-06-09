# -*- coding: utf-8 -*-
# @Time    : 4/6/23 8:39 PM
# @Author  : LIANYONGXING
# @FileName: ViTImageClassificationModel.py

import pytorch_lightning as pl
from transformers import AdamW
import torch.nn as nn
from model.vit import VitClsModel
import torch.nn.functional as F
import torch
from dataset.basic_datasets import build_dataloader, build_test_dataloader
from sklearn.metrics import classification_report
import argparse
from types import SimpleNamespace


class ViTImageClassificationModel(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        print(args)
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
        if isinstance(args, dict):
            args = SimpleNamespace(**args)
            args.mode = 'test'
        else:
            args.mode = 'train'

        self.args = args
        self.vit = VitClsModel(self.args.pretrain_path)
        if self.args.mode == 'train':
            self.get_dataloader()
        self.criterion = nn.CrossEntropyLoss()

        # self.vit = VitClsModel("/Users/user/Desktop/model_file/vit-base-patch16-224")

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)

        loss = self.criterion(logits, labels)
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
        # loss, accuracy = self.common_step(batch, batch_idx)
        # self.log("validation_loss", loss, on_epoch=True)
        # self.log("validation_accuracy", accuracy, on_epoch=True)
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)

        loss = self.criterion(logits, labels)

        predict_scores = F.softmax(logits, dim=1)
        predict_labels = torch.argmax(predict_scores, dim=-1)

        cls_report = classification_report(labels.cpu(),predict_labels.cpu(), output_dict=True)
        try:
            cls_report_bcase = cls_report['1']
            tf_board_logs = {
                "valid_loss": loss,
                "valid_acc": cls_report_bcase['precision'],
                "valid_recall": cls_report_bcase['recall'],
                "valid_f1": cls_report_bcase['f1-score']
            }
        except Exception as e:
            print(cls_report)
            tf_board_logs = {
                "valid_loss": loss,
                "valid_acc": 0,
                "valid_recall": 0,
                "valid_f1": 0
            }
        self.log_dict(tf_board_logs)
        return {'loss': loss, 'log': tf_board_logs}

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        return loss

    def predict_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        logits = self(pixel_values)
        predict_scores = F.softmax(logits, dim=1)
        predict_labels = torch.argmax(predict_scores, dim=-1)
        return predict_scores, predict_labels

    def configure_optimizers(self):
        # We could make the optimizer more fancy by adding a scheduler and specifying which parameters do
        # not require weight_decay but just using AdamW out-of-the-box works fine
        return AdamW(self.parameters(), lr=5e-5)

    def get_dataloader(self):
        self.train_dl, self.valid_dl = build_dataloader('', batch_size=self.args.batch_size,
                                                        vit_path=self.args.pretrain_path)

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.valid_dl

    def get_test_dataloader(self, path):
        test_loader = build_test_dataloader(path, batch_size=64, vit_path=self.args.pretrain_path)
        return test_loader

