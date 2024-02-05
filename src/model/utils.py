import torch

def opt_name_to_type(opt_name):
    if opt_name == 'Adam':
        return torch.optim.Adam
    if opt_name == 'Adagrad':
        return torch.optim.Adagrad
    if opt_name == 'SGD':
        return torch.optim.SGD
    raise ValueError(f"Unrecognized optimizer: {opt_name}")