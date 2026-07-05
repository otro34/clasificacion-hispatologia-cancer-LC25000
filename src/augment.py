"""Data augmentation dirigida

  - Geometricas (flips H/V, rotaciones de 90): el tejido en un parche no tiene
    una orientacion canonica, por lo que estas transformaciones son seguras y
    multiplican efectivamente el dataset.
  - Fotometricas (ColorJitter sobre brillo/contraste/saturacion/hue): simulan
    la variabilidad del tenido Hematoxilina-Eosina (H&E) entre laboratorios y
    escaneres, haciendo el modelo robusto a cambios de coloracion.
  - Val/Test NO llevan augmentation: solo resize + normalizacion, para una
    evaluacion deterministica y comparable.
"""
from torchvision import transforms

from . import config as C


def build_train_transforms():
    """Transforms con augmentation (solo para entrenamiento)."""
    return transforms.Compose([
        transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        # Rotaciones multiplos de 90 -> validas por la falta de orientacion canonica
        transforms.RandomChoice([
            transforms.RandomRotation((0, 0)),
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((270, 270)),
        ]),
        # Simula variabilidad de tenido H&E
        transforms.ColorJitter(brightness=0.15, contrast=0.15,
                               saturation=0.15, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])


def build_eval_transforms():
    """Transforms deterministicos para validacion y test."""
    return transforms.Compose([
        transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])


def denormalize(tensor):
    """Revierte la normalizacion ImageNet para poder visualizar una imagen."""
    import torch
    mean = torch.tensor(C.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(C.IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)
