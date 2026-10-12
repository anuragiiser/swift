"""CIFAR-100 super-class data for Swift (ResNet-18, 20-way).

The Original model is a 20-way super-class classifier: the 100 fine labels are
mapped to their coarse super-class, so the forget request ("aquarium fish")
names a fine class whose *label* is shared with the four other fine classes of
its super-class ("fish"). That shared label is what makes the retain set
entangled with the forget set, and it is what D_f accuracy is measured against.

Cheng et al. (2026) use the same construction; the data module their public
release omits is reconstructed here (super-class mapping, normalisation
constants and the 224-pixel resize that the ImageNet-pretrained ResNet-18
expects).

torchvision downloads CIFAR-100 on first use, so nothing has to be prepared.
"""
import os

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

N_SUPERCLASSES = 20

# Fine label (0..99) -> super-class (0..19), the standard CIFAR-100 mapping.
# Fine label 1 ("aquarium fish") sits in super-class 1 ("fish").
SUPERCLASS_MAPPING = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]

SUPERCLASS_NAMES = (
    "aquatic mammals", "fish", "flowers", "food containers",
    "fruit and vegetables", "electrical devices", "furniture", "insects",
    "large carnivores", "large man-made outdoor things",
    "large natural outdoor scenes", "large omnivores and herbivores",
    "medium-sized mammals", "non-insect invertebrates", "people", "reptiles",
    "small mammals", "trees", "vehicles 1", "vehicles 2",
)

# The four forget classes reported in the paper, one per super-class.
FINE_NAMES = {1: "aquarium fish", 4: "beaver", 6: "bee", 8: "bicycle"}

CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])


class SuperClassDataset(Dataset):
    """Wraps CIFAR-100 (or a Subset of it) and returns coarse labels."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        image, fine_label = self.base[i]
        return image, SUPERCLASS_MAPPING[fine_label]


def cifar100_base(train, data_root="~/data", download=True, transform=None):
    """The raw torchvision dataset, with fine labels in `.targets`."""
    return datasets.CIFAR100(root=os.path.expanduser(data_root), train=train,
                             download=download,
                             transform=DEFAULT_TRANSFORM if transform is None else transform)


def coarse_targets(base):
    return np.asarray(SUPERCLASS_MAPPING)[np.asarray(base.targets)]


def make_loader(base, indices, batch_size, shuffle=False, num_workers=8,
                drop_last=False):
    """Loader over a subset of `base`, relabelled to super-classes."""
    drop_last = drop_last and len(indices) > batch_size
    dataset = SuperClassDataset(Subset(base, np.asarray(indices).tolist()))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=drop_last)


def fine_name(fine_label):
    return FINE_NAMES.get(fine_label, f"fine class {fine_label}")


def superclass_name(fine_label):
    return SUPERCLASS_NAMES[SUPERCLASS_MAPPING[fine_label]]
