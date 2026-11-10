"""Metric helpers."""

import torch


def accuracy(pred, target):
    return (pred.argmax(dim=-1) == target).float().mean().item()


def masked_mae(pred, target, mask=None):
    err = (pred - target).abs()
    if mask is not None:
        err = err * mask
        return err.sum().item() / mask.sum().item()
    return err.mean().item()
