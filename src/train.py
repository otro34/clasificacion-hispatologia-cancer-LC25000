"""Loop de entrenamiento con early stopping y checkpoints.

Funciona en CUDA / MPS / CPU sin cambios (usa C.DEVICE).
"""
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm  # version de texto: se guarda como output plano (no widget)

from . import config as C


def _run_epoch(model, loader, criterion, optimizer=None):
    """Una epoca. Si optimizer es None -> modo evaluacion."""
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, all_preds, all_targets = 0.0, [], []
    for imgs, targets in tqdm(loader, leave=False,
                              desc="train" if train_mode else "eval"):
        imgs = imgs.to(C.DEVICE)
        targets = targets.to(C.DEVICE)

        with torch.set_grad_enabled(train_mode):
            logits = model(imgs)
            loss = criterion(logits, targets)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        all_preds.append(logits.argmax(1).detach().cpu())
        all_targets.append(targets.detach().cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    avg_loss = total_loss / len(loader.dataset)
    acc = (preds == targets).mean()
    f1 = f1_score(targets, preds, average="macro")
    return avg_loss, acc, f1


def train_model(model, loaders, epochs, lr, tag="model",
                weight_decay=None, patience=None):
    """Entrena `model`. Devuelve (mejor_modelo, historial).

    Guarda el mejor checkpoint (por F1 de validacion) en outputs/checkpoints/.
    """
    weight_decay = C.WEIGHT_DECAY if weight_decay is None else weight_decay
    patience = C.EARLY_STOPPING_PATIENCE if patience is None else patience

    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2)

    history = {"train_loss": [], "val_loss": [], "train_acc": [],
               "val_acc": [], "train_f1": [], "val_f1": []}
    best_f1, best_state, epochs_no_improve = -1.0, None, 0
    ckpt_path = C.CHECKPOINTS_DIR / f"{tag}_best.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = _run_epoch(model, loaders["train"], criterion, optimizer)
        va_loss, va_acc, va_f1 = _run_epoch(model, loaders["val"], criterion, None)
        scheduler.step(va_f1)

        history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc); history["val_acc"].append(va_acc)
        history["train_f1"].append(tr_f1); history["val_f1"].append(va_f1)

        print(f"[{tag}] Epoch {epoch:02d}/{epochs} "
              f"({time.time()-t0:.0f}s) "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.3f} f1={tr_f1:.3f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.3f} f1={va_f1:.3f}")

        if va_f1 > best_f1:
            best_f1, epochs_no_improve = va_f1, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[{tag}] Early stopping en epoca {epoch} "
                      f"(mejor val_f1={best_f1:.3f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[{tag}] Mejor val_f1={best_f1:.3f} | checkpoint: {ckpt_path}")
    return model, history


def two_stage_training(model, loaders, tag="efficientnet_b0"):
    """Entrenamiento completo en 2 etapas (feature extraction + fine-tuning)."""
    from .model import freeze_backbone, unfreeze_all, trainable_parameters

    print("=== Etapa A: feature extraction (backbone congelado) ===")
    freeze_backbone(model)
    tr, tot = trainable_parameters(model)
    print(f"Parametros entrenables: {tr:,} / {tot:,}")
    model, hist_a = train_model(model, loaders, C.EPOCHS_HEAD, C.LR_HEAD,
                                tag=f"{tag}_head")

    print("\n=== Etapa B: fine-tuning (backbone descongelado) ===")
    unfreeze_all(model)
    tr, tot = trainable_parameters(model)
    print(f"Parametros entrenables: {tr:,} / {tot:,}")
    model, hist_b = train_model(model, loaders, C.EPOCHS_FINETUNE, C.LR_FINETUNE,
                                tag=f"{tag}_finetune")

    # Concatenar historiales para graficar curvas continuas
    history = {k: hist_a[k] + hist_b[k] for k in hist_a}
    history["stage_boundary"] = len(hist_a["train_loss"])
    return model, history
