import numpy as np
import torch
import torch.utils.data as Tdata
import torchvision.datasets as TVdatasets
from torchvision import transforms as TVtransforms
import pathlib
import os
from typing import List, Tuple, Union, Any
import itertools

#TODO: work properly with random seed!!!

file_dir = pathlib.Path(__file__).parent.resolve()

DEFAULT_DATASET_PATH = os.path.join(file_dir, 'data/cmnist')

def random_color(im):
    hue = 360*np.random.rand()
    d = (im *(hue%60)/60)
    im_min, im_inc, im_dec = torch.zeros_like(im), d, im - d
    H = round(hue/60) % 6    
    cmap = [[0, 3, 2], [2, 0, 3], [1, 0, 3], [1, 2, 0], [3, 1, 0], [0, 1, 2]]
    return torch.cat((im, im_min, im_dec, im_inc), dim=0)[cmap[H]]


class CMNISTDataset(Tdata.Dataset):

    def __init__(
        self,
        train=True,
        digit='two',
        spat_dim=(28, 28),
        root=DEFAULT_DATASET_PATH,
        download=False,
        pix_range=(0., 1.)
    ) -> None:
        super().__init__()
        self.digit = digit
        assert digit in ['two', 'three']
        digit_map = {
            'two': 2,
            'three': 3
        }
        _m, _std = pix_range[0]/float(pix_range[0] - pix_range[1]), 1./float(pix_range[1] - pix_range[0])
        TRANSFORM = TVtransforms.Compose([
            TVtransforms.Resize(spat_dim),
            TVtransforms.ToTensor(),
            random_color,
            TVtransforms.Normalize([_m],[_std])
        ])
        mnist = TVdatasets.MNIST(root=root, train=train, download=download, transform=TRANSFORM)
        idx = np.array(range(len(mnist)))
        mnist_digit = Tdata.Subset(mnist, idx[mnist.targets==digit_map[digit]])
        self.mnist_digit = mnist_digit

    def __len__(self) -> int:
        return len(self.mnist_digit)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.mnist_digit[idx][0]

class CMNISTPairedDataset(Tdata.Dataset):

    def __init__(
        self,
        train : bool = True,
        digits : Tuple[str, str] = ('two', 'three'),
        spat_dim : Tuple[int, int] = (28, 28),
        root : str =DEFAULT_DATASET_PATH,
        download : bool = False,
        pix_range : Tuple[float, float] = (0., 1.),
        dummy_class : bool = True
    ) -> None:
        self.dataset1 = CMNISTDataset(train, digits[0], spat_dim, root, download, pix_range)
        self.dataset2 = CMNISTDataset(train, digits[1], spat_dim, root, download, pix_range)
        self.order = np.array(list(itertools.product(
            range(len(self.dataset1)), 
            range(len(self.dataset2)))))
        self.dummy_class = dummy_class
        super().__init__()
    
    def __len__(self) -> int:
        return len(self.dataset1) * len(self.dataset2)
    
    def __getitem__(self, index: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        i1, i2 = self.order[index]
        x, y = self.dataset1[i1], self.dataset2[i2]
        res = torch.cat((x, y), 0)
        if not self.dummy_class:
            return res
        return res, torch.zeros(1, dtype=torch.long, device=res.device)
        

