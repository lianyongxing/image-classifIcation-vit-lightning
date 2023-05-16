# -*- coding: utf-8 -*-
# @Time    : 4/6/23 5:44 PM
# @Author  : LIANYONGXING
# @FileName: basic_datasets.py
from datasets import load_dataset
from PIL import Image
import torch
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)
from transformers import ViTImageProcessor
from torch.utils.data import DataLoader


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def image_dataset_transform_process(processor, trainset, validset):
    image_mean = processor.image_mean
    image_std = processor.image_std
    size = processor.size["height"]
    print(image_mean, image_std, size)
    normalize = Normalize(mean=image_mean, std=image_std)

    _train_transforms = Compose(
            [
                RandomResizedCrop(size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )

    _val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )

    def train_transforms(examples):
        examples['img'] = [Image.open(url) for url in examples['url']]
        examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
        return examples

    def val_transforms(examples):
        examples['img'] = [Image.open(url) for url in examples['url']]
        examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
        return examples

    trainset.set_transform(train_transforms)
    validset.set_transform(val_transforms)
    return trainset, validset


def build_dataloader(fp, batch_size, vit_path):
    processor = ViTImageProcessor.from_pretrained(vit_path)

    train_ds_raw = load_dataset('csv', data_dir='/data/yxlian/leaderface/', data_files='trainset_leaderface01.csv',
                                split='train')
    valid_ds_raw = load_dataset('csv', data_dir='/data/yxlian/leaderface/', data_files='testset_leaderface01.csv',
                                split='train')

    train_ds, valid_ds = image_dataset_transform_process(processor, train_ds_raw, valid_ds_raw)

    print('train_ds num: %s' % len(train_ds))
    print('valid_ds num: %s' % len(valid_ds))

    train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_ds, collate_fn=collate_fn, batch_size=batch_size)

    return train_dataloader, valid_dataloader


if __name__ == '__main__':

    batch_size = 4
    train_dl, valid_dl = build_dataloader('/Users/user/Desktop/model_ViT/testdats.csv', batch_size)

    batch = next(iter(train_dl))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)

    assert batch['pixel_values'].shape == (batch_size, 3, 224, 224)
    assert batch['labels'].shape == (batch_size,)
    print(next(iter(valid_dl))['pixel_values'].shape)