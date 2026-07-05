"""Construccion del modelo con transfer learning (timm).

Estrategia en dos etapas:
  - Etapa A (feature extraction): backbone congelado, se entrena solo la cabeza.
  - Etapa B (fine-tuning): se descongelan las capas superiores con LR bajo.

Entrada : tensor RGB (B, 3, 224, 224) normalizado con stats de ImageNet.
Salida  : logits (B, NUM_CLASSES) -> softmax para probabilidades.
"""
import timm
import torch.nn as nn

from . import config as C


def build_model(model_name: str = None, pretrained: bool = True):
    """Crea un modelo preentrenado con la cabeza adaptada a NUM_CLASSES."""
    model_name = model_name or C.MODEL_NAME
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=C.NUM_CLASSES,  # timm reemplaza la cabeza automaticamente
    )
    return model.to(C.DEVICE)


def freeze_backbone(model):
    """Etapa A: congela todo y deja entrenable solo la cabeza de clasificacion."""
    for param in model.parameters():
        param.requires_grad = False

    classifier = model.get_classifier()  # API de timm
    for param in classifier.parameters():
        param.requires_grad = True
    return model


def unfreeze_all(model):
    """Etapa B: descongela todos los parametros para fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    return model


def trainable_parameters(model):
    """Cuenta parametros entrenables vs totales (para reportar en la presentacion)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
