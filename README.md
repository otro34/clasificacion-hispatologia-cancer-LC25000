# Clasificacion de cancer en imagenes histopatologicas (LC25000)

Transfer learning con **EfficientNet-B0** (y comparacion con ResNet50) para clasificar
parches histopatologicos en 5 clases (colon y pulmon, benigno/maligno).

- Notebook principal: [`notebooks/histopatologia_cancer.ipynb`](notebooks/histopatologia_cancer.ipynb)

## Corre en local (Mac/MPS) y en Colab (CUDA)

El codigo autodetecta el dispositivo (`cuda` > `mps` > `cpu`) en `src/config.py`.

### Local (Mac Apple Silicon, ya configurado)

```bash
source .venv/bin/activate

jupyter lab            # o: jupyter notebook
```

### Colab

1. Subir la carpeta `src/` (o clona el repo) y el notebook.
2. Descomentar la celda de `pip install` en el setup.
3. Subir tu `kaggle.json` para descargar el dataset.

## Dataset

LC25000 - Kaggle: `andrewmvd/lung-and-colon-cancer-histopathological-images`
(25.000 imagenes, 5 clases balanceadas de 5.000 c/u, 768x768).

Descargar con la API de Kaggle (ver celda 1.1 del notebook). La carpeta
`lung_colon_image_set/` debe quedar bajo `data/` (local) o `/content/` (Colab),
segun `C.DATA_DIR`.

En este repo estamos incluyendo las imágenes del dataset para simplificar la ejecución.

## Estructura

```
src/
  config.py     # dispositivo, rutas, hiperparametros
  data.py       # scan + split estratificado + dataloaders
  augment.py    # data augmentation dirigida (H&E)
  model.py      # transfer learning (timm)
  train.py      # entrenamiento 2 etapas + early stopping
  evaluate.py   # metricas, matriz confusion, Grad-CAM
notebooks/
  histopatologia_cancer.ipynb
outputs/
  checkpoints/  # mejores modelos (.pt)
  figures/      # graficas para la presentacion
```
