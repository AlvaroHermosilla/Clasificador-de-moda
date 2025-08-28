import tarfile
import urllib.request
import os
import shutil
from PIL import Image


from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_curve,
                             roc_auc_score,
                             log_loss,
                             classification_report,
                             confusion_matrix,
                             ConfusionMatrixDisplay,
                            )
from sklearn.preprocessing import label_binarize

import torch
import random
import numpy as np

def dowload(url):
    tar_filename = "images-dataSAT.tar"
    extracted_folder = "images-dataSAT"
    # Descarga.
    urllib.request.urlretrieve(url, tar_filename)
    print(f"Descarga de {tar_filename} completa.")

    # Comprobamos si la carpeta existe.
    if os.path.exists(extracted_folder):
        shutil.rmtree(extracted_folder)  # Elimina la carpeta si ya existe
        print(f"Carpeta {extracted_folder} eliminada y spbreescrita.")

    # Extracción del archivo tar.
    with tarfile.open(tar_filename, "r") as tar:
        tar.extractall(path=extracted_folder)
        print(f"Archivo {tar_filename} extraído en la carpeta {extracted_folder}.")



def model_metrics(y_true, y_pred, y_prob, class_labels):
    metrics = {'Accuracy': accuracy_score(y_true, y_pred),
               'Precision': precision_score(y_true, y_pred),
               'Recall': recall_score(y_true, y_pred),
               'Loss': log_loss(y_true, y_prob),
               'F1 Score': f1_score(y_true, y_pred),
               'ROC-AUC': roc_auc_score(y_true, y_prob),
               'Confusion Matrix': confusion_matrix(y_true, y_pred),
               'Classification Report': classification_report(y_true, y_pred, target_names=class_labels, digits=4),
               "Class labels": class_labels
              }
    return metrics


def print_metrics(y_true, y_pred, y_prob, class_labels, model_name):
    metrics = model_metrics(y_true, y_pred, y_prob, class_labels)
    print(f"Evaluation metrics for the \033[1m{model_name}\033[0m")
    print(f"Accuracy: {'':<1}{metrics["Accuracy"]:.4f}")
    print(f"ROC-AUC: {'':<2}{metrics["ROC-AUC"]:.4f}")
    print(f"Loss: {'':<5}{metrics["Loss"]:.4f}\n")
    print(f"Classification report:\n\n  {metrics["Classification Report"]}")
    print("========= Confusion Matrix =========")
    disp = ConfusionMatrixDisplay(confusion_matrix=metrics["Confusion Matrix"],
                                  display_labels=metrics["Class labels"])


def worker_init_fn(worker_id: int) -> None:
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)