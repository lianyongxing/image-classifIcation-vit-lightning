# -*- coding: utf-8 -*-
# @Time    : 4/6/23 8:46 PM
# @Author  : LIANYONGXING
# @FileName: train.py


from task.image_classification_vit_task import ViTImageClassificationModel
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import os
import json


def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--pretrain_path", default='google/vit-base-patch16-224-in21k', type=str,
                        help="bert config file")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--epochs", type=int, default=2, help="max epochs")
    parser.add_argument("--save_every_epoch", type=int, default=1, help="save_every_epoch")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--use_memory", action="store_true", help="load dataset to memory to accelerate.")
    parser.add_argument("--train_filepath", required=True, type=str, help="train data path")
    parser.add_argument("--save_path", required=True, type=str, help="train data path")
    parser.add_argument("--save_topk", default=2, type=int, help="save topk checkpoint")
    parser.add_argument("--warmup_proportion", default=0.01, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float, help="dropout probability")
    parser.add_argument("--tag", default='v001', type=str, help="version")
    parser.add_argument("--mode", default='train', type=str, help="version")
    parser.add_argument("--gpu_num", default=0, type=int, help="use gpu num")
    return parser


if __name__ == '__main__':

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # save args
    save_args_folder = os.path.join(args.save_path, args.tag)
    if os.path.exists(save_args_folder):
        print('tag版本号已经存在，请检查')
        exit(0)
    else:
        os.makedirs(save_args_folder)

    logger = TensorBoardLogger(
        save_dir=save_args_folder,
        name='log'
    )

    checkpoint_callback = ModelCheckpoint(dirpath=save_args_folder,
                                          every_n_epochs=args.save_every_epoch,
                                          save_on_train_epoch_end=True,
                                          save_top_k=-1)

    with open(os.path.join(save_args_folder, "args.json"), 'w') as f:
        args_dict = args.__dict__
        args_dict = {k: v for k, v in args_dict.items() if v is not None}
        json.dump(args_dict, f, indent=4)

    model = ViTImageClassificationModel(args)

    trainer = Trainer.from_argparse_args(args,
                                         max_epochs=args.epochs,
                                         gpus=args.gpu_num,
                                         logger=logger,
                                         callbacks=checkpoint_callback,
                                         )

    trainer.fit(model)