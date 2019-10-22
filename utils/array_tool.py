"""
tools to convert specified type
"""
import torch as t
import numpy as np


def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, t.Tensor):
        return data.detach().cpu().numpy()
    else:
        raise Exception("No tonumpy conversion for type {}".format(type(data)))

def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = t.from_numpy(data)
    elif isinstance(data, t.Tensor):
        tensor = data.detach()
    else:
        raise Exception("No totensor conversion for type {}".format(type(data)))
    if cuda:
        tensor = tensor.cuda()
    return tensor


def scalar(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    elif isinstance(data, t.Tensor):
        return data.item()
    elif isinstance(data, float):
        return data
    else:
        raise Exception("No scalar conversion for type {}".format(type(data)))