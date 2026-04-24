from itertools import product

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, RandomHorizontalFlip,
                                    RandomPerspective, RandomRotation, Resize,
                                    ToTensor)
from torchvision.transforms.transforms import RandomResizedCrop

from datasets.read_datasets import load_composition_datasetdict

BICUBIC = InterpolationMode.BICUBIC
n_px = 224


def transform_image(split="train", imagenet=False):
    if imagenet:
        # from czsl repo.
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = Compose(
            [
                RandomResizedCrop(n_px),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(
                    mean,
                    std,
                ),
            ]
        )
        return transform

    if split == "test" or split == "val":
        transform = Compose(
            [
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    else:
        transform = Compose(
            [
                # RandomResizedCrop(n_px, interpolation=BICUBIC),
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                RandomHorizontalFlip(),
                RandomPerspective(),
                RandomRotation(degrees=5),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    return transform

def _hf_image_to_pil_rgb(img):
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, dict):
        from io import BytesIO

        b = img.get("bytes")
        if b is not None:
            return Image.open(BytesIO(b)).convert("RGB")
        p = img.get("path")
        if p:
            return Image.open(p).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img).__name__}")


class CompositionDataset(Dataset):
    def __init__(
            self,
            dataset_name,
            phase,
            split='compositional-split-natural',
            open_world=False,
            imagenet=False
            # inductive=True
    ):
        self.dataset_name = dataset_name
        self.phase = phase
        self.split = split
        self.open_world = open_world

        # new addition
        # if phase == 'train':
        #     self.inductive = inductive
        # else:
        #     self.inductive = False

        self.feat_dim = None
        self.transform = transform_image(phase, imagenet=imagenet)
        self.ds_dict = load_composition_datasetdict(dataset_name)

        self.attrs, self.objs, self.pairs, \
                self.train_pairs, self.val_pairs, \
                self.test_pairs = self.parse_split()

        if self.open_world:
            self.pairs = list(product(self.attrs, self.objs))

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        else:
            self.data = self.test_data

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        print('# train pairs: %d | # val pairs: %d | # test pairs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('# train images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))

        self.train_pair_to_idx = dict(
            [(pair, idx) for idx, pair in enumerate(self.train_pairs)]
        )

        if self.open_world:
            mask = [1 if pair in set(self.train_pairs) else 0 for pair in self.pairs]
            self.seen_mask = torch.BoolTensor(mask) * 1.

            self.obj_by_attrs_train = {k: [] for k in self.attrs}
            for (a, o) in self.train_pairs:
                self.obj_by_attrs_train[a].append(o)

            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.attrs_by_obj_train = {k: [] for k in self.objs}
            for (a, o) in self.train_pairs:
                self.attrs_by_obj_train[o].append(a)

    def get_split_info(self):
        train_data, val_data, test_data = [], [], []
        split_map = {
            "train": train_data,
            "val": val_data,
            "test": test_data,
        }
        for split_name, bucket in split_map.items():
            if split_name not in self.ds_dict:
                continue
            for row in self.ds_dict[split_name]:
                attr = str(row["attr"])
                obj = str(row["obj"])
                if attr == "NA" or (attr, obj) not in self.pairs:
                    continue
                bucket.append([row["image"], attr, obj])
        return train_data, val_data, test_data

    def parse_split(self):
        def pairs_from_split(name):
            if name not in self.ds_dict:
                return []
            pairs = []
            for row in self.ds_dict[name]:
                pairs.append((str(row["attr"]), str(row["obj"])))
            return sorted(set(pairs))

        tr_pairs = pairs_from_split("train")
        vl_pairs = pairs_from_split("val")
        ts_pairs = pairs_from_split("test")
        tr_attrs, tr_objs = zip(*tr_pairs) if tr_pairs else ((), ())
        vl_attrs, vl_objs = zip(*vl_pairs) if vl_pairs else ((), ())
        ts_attrs, ts_objs = zip(*ts_pairs) if ts_pairs else ((), ())

        all_attrs, all_objs = sorted(
            list(set(list(tr_attrs) + list(vl_attrs) + list(ts_attrs)))), sorted(
                list(set(list(tr_objs) + list(vl_objs) + list(ts_objs))))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def __getitem__(self, index):
        image, attr, obj = self.data[index]
        img = _hf_image_to_pil_rgb(image)
        img = self.transform(img)

        if self.phase == 'train':
            data = [
                img, self.attr2idx[attr], self.obj2idx[obj], self.train_pair_to_idx[(attr, obj)]
            ]
        else:
            data = [
                img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]
            ]

        return data

    def __len__(self):
        return len(self.data)
