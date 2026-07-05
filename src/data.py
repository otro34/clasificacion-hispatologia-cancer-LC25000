"""Carga del dataset LC25000, split estratificado y DataLoaders.

Estructura esperada del dataset (tras descargar y descomprimir):

    DATA_DIR/
        colon_image_sets/
            colon_n/    *.jpeg
            colon_aca/  *.jpeg
        lung_image_sets/
            lung_n/     *.jpeg
            lung_aca/   *.jpeg
            lung_scc/   *.jpeg

El dataset original de Kaggle (LC25000) trae justamente esa organizacion.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from . import config as C
from .augment import build_train_transforms, build_eval_transforms


# Mapeo nombre de clase -> subcarpeta relativa dentro de DATA_DIR
CLASS_DIRS = {
    "colon_n": "colon_image_sets/colon_n",
    "colon_aca": "colon_image_sets/colon_aca",
    "lung_n": "lung_image_sets/lung_n",
    "lung_aca": "lung_image_sets/lung_aca",
    "lung_scc": "lung_image_sets/lung_scc",
}


class HistoDataset(Dataset):
    """Dataset de parches histopatologicos a partir de listas de rutas."""

    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx]


def scan_dataset(data_dir: Path = None):
    """Recorre las carpetas y devuelve (paths, labels, label_to_idx).

    labels son enteros segun el orden de C.CLASSES.
    """
    data_dir = Path(data_dir or C.DATA_DIR)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"No se encontro el dataset en {data_dir}. "
            "Descargalo primero (ver README / notebook, seccion descarga)."
        )

    label_to_idx = {cls: i for i, cls in enumerate(C.CLASSES)}
    paths, labels = [], []
    for cls, rel in CLASS_DIRS.items():
        folder = data_dir / rel
        if not folder.exists():
            raise FileNotFoundError(f"Falta la carpeta de clase: {folder}")
        files = sorted(
            [p for p in folder.iterdir()
             if p.suffix.lower() in (".jpeg", ".jpg", ".png")]
        )
        paths.extend(files)
        labels.extend([label_to_idx[cls]] * len(files))

    return paths, np.array(labels), label_to_idx


def stratified_split(paths, labels, seed=None):
    """Split estratificado 70/15/15 manteniendo proporcion de clases."""
    seed = seed or C.SEED
    paths = np.array(paths)

    # Primero separamos test (15%)
    p_trainval, p_test, y_trainval, y_test = train_test_split(
        paths, labels, test_size=C.TEST_SPLIT,
        stratify=labels, random_state=seed,
    )
    # Del resto, val proporcional (15% del total -> sobre el 85% restante)
    val_rel = C.VAL_SPLIT / (C.TRAIN_SPLIT + C.VAL_SPLIT)
    p_train, p_val, y_train, y_val = train_test_split(
        p_trainval, y_trainval, test_size=val_rel,
        stratify=y_trainval, random_state=seed,
    )
    return (p_train, y_train), (p_val, y_val), (p_test, y_test)


def build_dataloaders(data_dir: Path = None, batch_size: int = None):
    """Pipeline completo: scan -> split -> Datasets -> DataLoaders.

    Devuelve un dict con loaders y los splits crudos (utiles para EDA/Grad-CAM).
    """
    batch_size = batch_size or C.BATCH_SIZE
    paths, labels, label_to_idx = scan_dataset(data_dir)
    (p_tr, y_tr), (p_va, y_va), (p_te, y_te) = stratified_split(paths, labels)

    train_ds = HistoDataset(p_tr, y_tr, build_train_transforms())
    val_ds = HistoDataset(p_va, y_va, build_eval_transforms())
    test_ds = HistoDataset(p_te, y_te, build_eval_transforms())

    common = dict(num_workers=C.NUM_WORKERS, pin_memory=C.PIN_MEMORY)
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common),
        "splits": {
            "train": (p_tr, y_tr),
            "val": (p_va, y_va),
            "test": (p_te, y_te),
        },
        "label_to_idx": label_to_idx,
    }


def class_counts(labels):
    """Conteo por clase (para EDA). Devuelve dict nombre_clase -> n."""
    counts = {}
    for i, cls in enumerate(C.CLASSES):
        counts[C.CLASS_NAMES[cls]] = int((np.array(labels) == i).sum())
    return counts
