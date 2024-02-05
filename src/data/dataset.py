import numpy as np
import torch
from torch.utils.data import Dataset
import sklearn

from ..config import Config
from .utils import std_scale, std_scale_inv

def generate_gauss_2d_dataset(mu, sigma, n_samples):
    samples = np.random.normal(
        loc=mu, scale=sigma, size=(n_samples, 2)).astype(np.float32)
    return samples


def generate_swissroll(scale, noise, n_samples):
    samples, _ = sklearn.datasets.make_swiss_roll(
        n_samples, noise=noise, random_state=Config.SEED)
    return samples[:, (0, 2)].astype(np.float32)*scale


class GaussDataset(Dataset):
    def __init__(self, mu=0, sigma=1, n_samples=1000, sample_scaled=Config.SCALED_REGIME):
        super().__init__()
        self.data = torch.from_numpy(
            generate_gauss_2d_dataset(mu, sigma, n_samples)).to(Config.DEVICE)

        self.mu = self.data.mean(dim=0)
        self.std = self.data.std(dim=0)

        self.sample_scaled = sample_scaled

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

    def sample(self, n):
        sample = torch.stack(
            list(self.data[np.random.randint(low=0, high=len(self.data))]
                 for _ in range(n))
        )
        if self.sample_scaled:
            sample = std_scale(sample, self.mu, self.std)
        return sample


class SwissrollDataset(Dataset):
    def __init__(self, scale=1, noise=0.75, n_samples=1000, sample_scaled=Config.SCALED_REGIME):
        super().__init__()
        self.data = torch.from_numpy(generate_swissroll(
            scale, noise, n_samples)).to(Config.DEVICE)

        self.mu = self.data.mean(dim=0)
        self.std = self.data.std(dim=0)

        self.sample_scaled = sample_scaled

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

    def sample(self, n):
        sample = torch.stack(
            list(self.data[np.random.randint(
                low=0, high=len(self.data), size=n)])
        )
        if self.sample_scaled:
            sample = std_scale(sample, self.mu, self.std)
        return sample
