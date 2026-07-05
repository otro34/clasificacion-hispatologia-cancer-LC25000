"""Evaluacion: metricas, matriz de confusion, curvas y Grad-CAM."""
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
)

from . import config as C


@torch.no_grad()
def predict(model, loader):
    """Devuelve (y_true, y_pred, y_prob) sobre un loader."""
    model.eval()
    ys, ps, probs = [], [], []
    for imgs, targets in loader:
        imgs = imgs.to(C.DEVICE)
        logits = model(imgs)
        prob = torch.softmax(logits, dim=1)
        ps.append(prob.argmax(1).cpu())
        probs.append(prob.cpu())
        ys.append(targets)
    return (torch.cat(ys).numpy(),
            torch.cat(ps).numpy(),
            torch.cat(probs).numpy())


def class_labels():
    """Etiquetas legibles en el orden de C.CLASSES."""
    return [C.CLASS_NAMES[c] for c in C.CLASSES]


def report(y_true, y_pred):
    """Imprime classification report y devuelve dict de metricas clave."""
    labels = class_labels()
    txt = classification_report(y_true, y_pred, target_names=labels, digits=4)
    print(txt)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "report_text": txt,
    }


def plot_confusion(y_true, y_pred, normalize=False, save_as=None, title=None):
    """Grafica la matriz de confusion (absoluta o normalizada)."""
    labels = class_labels()
    cm = confusion_matrix(y_true, y_pred)
    fmt = "d"
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar=True, square=True)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title(title or ("Matriz de confusion"
                           + (" (normalizada)" if normalize else "")))
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_as:
        fig.savefig(C.FIGURES_DIR / save_as, dpi=150, bbox_inches="tight")
    return fig


def plot_history(history, save_as=None):
    """Grafica curvas de loss / accuracy / f1 (train vs val)."""
    boundary = history.get("stage_boundary")
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, key, name in zip(
        axes, ["loss", "acc", "f1"], ["Loss", "Accuracy", "F1 macro"]
    ):
        ax.plot(epochs, history[f"train_{key}"], label="train", marker="o")
        ax.plot(epochs, history[f"val_{key}"], label="val", marker="s")
        if boundary:
            ax.axvline(boundary + 0.5, color="gray", ls="--", alpha=0.7,
                       label="inicio fine-tuning")
        ax.set_xlabel("Epoca"); ax.set_ylabel(name); ax.set_title(name)
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_as:
        fig.savefig(C.FIGURES_DIR / save_as, dpi=150, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------------
# Grad-CAM (interpretabilidad)
# ----------------------------------------------------------------------------
def gradcam_on_samples(model, images, target_layer, n=6, save_as=None):
    """Genera mapas Grad-CAM para `images` (tensores ya normalizados).

    `target_layer`: capa convolucional objetivo. Para EfficientNet-B0 en timm
    suele ser `model.conv_head`; para ResNet50 `model.layer4[-1]`.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from .augment import denormalize

    model.eval()
    cam = GradCAM(model=model, target_layers=[target_layer])

    n = min(n, len(images))
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    for i in range(n):
        inp = images[i:i + 1].to(C.DEVICE)
        grayscale = cam(input_tensor=inp)[0]
        rgb = denormalize(images[i]).permute(1, 2, 0).numpy()
        overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)

        axes[0, i].imshow(rgb); axes[0, i].set_title("Original"); axes[0, i].axis("off")
        axes[1, i].imshow(overlay); axes[1, i].set_title("Grad-CAM"); axes[1, i].axis("off")
    plt.tight_layout()
    if save_as:
        fig.savefig(C.FIGURES_DIR / save_as, dpi=150, bbox_inches="tight")
    return fig
