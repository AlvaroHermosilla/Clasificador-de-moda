import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tqdm import tqdm
import os

from utils import dowload,model_metrics,print_metrics,worker_init_fn

#Formacion de pipelines.
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import random
import torch.optim as optim
import time
from torch.utils.data import Subset
from functools import partial

# Entrenamiento y test. 
import torch
import torch.nn as nn
from model import CNN_bn
import torch.nn.functional as F

# Metricas.
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

#Descarga de los datos.
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5vTzHBmQUaRNJQe5szCyKw/images-dataSAT.tar"
dowload = dowload(url)

#----------------------------------------
# Inicializamos las variables.
#----------------------------------------
seed = 1  
batch_size = 64
loss_history = {'train': [], 'val': []}
acc_history = {'train': [], 'val': []}
n_epochs = 3
model_name = 'Pytorch_CNN_Clasificator'
best_loss = float('inf')
# Definimos las rutas
extracted_folder = "images-dataSAT"
base_path = os.path.join(extracted_folder, 'images_dataSAT')
agri_path= os.path.join(base_path, 'class_1_agri')
non_agri_path = os.path.join(base_path, 'class_0_non_agri')
#----------------------------------------
# Formamos los pipelines.
#----------------------------------------
transform_train = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomVerticalFlip(p = 0.5),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])

transform_test = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])
dataset_imagefolder_train = ImageFolder(root=base_path, transform=transform_train)
dataset_imagefolder_test = ImageFolder(root=base_path, transform=transform_test)
dataset_size = len(dataset_imagefolder_train)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

# Índices para cada subset
indices = list(range(dataset_size))
random.seed(seed)
random.shuffle(indices)

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]


train_dataset = Subset(dataset_imagefolder_train, train_indices)
val_dataset = Subset(dataset_imagefolder_test, val_indices)
test_dataset = Subset(dataset_imagefolder_test, test_indices)


# Preparamos los Loaders
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0,
                          worker_init_fn=partial(worker_init_fn, seed))
val_loader = DataLoader(val_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=0,
                          worker_init_fn=partial(worker_init_fn, seed))
test_loader = DataLoader(test_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=0,
                          worker_init_fn=partial(worker_init_fn, seed))
#----------------------------------------
# Entrenamiento.
#----------------------------------------
model =CNN_bn(number_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for epoch in range(n_epochs):
    # Training Phase
    start_time = time.time() # to get the training time for each epoch
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0  # for the training metrics
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")):
        images, labels = images.to(device), labels.to(device)  # labels as integer class indices
        optimizer.zero_grad()
        outputs = model(images)  # outputs are raw logits
        loss = criterion(outputs, labels)  # criterion is nn.CrossEntropyLoss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    # Validation Phase
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0 #  for the validation metrics
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    # Save the best model
    avg_val_loss = val_loss/len(val_loader)
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), model_name)

    # Store metrics
    loss_history['train'].append(train_loss/len(train_loader))
    loss_history['val'].append(val_loss/len(val_loader))
    acc_history['train'].append(train_correct/train_total)
    acc_history['val'].append(val_correct/val_total)

    #print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {loss_history['train'][-1]:.4f} | Val Loss: {loss_history['val'][-1]:.4f}")
    print(f"Train Acc: {acc_history['train'][-1]:.4f} | Val Acc: {acc_history['val'][-1]:.4f}")
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1} training completed in {epoch_time:.2f} seconds\n")

#----------------------------------------
# Test.
#----------------------------------------
# Cargamos el modelo.
model_dir = 'Pytorch_CNN_Clasificator'
model_path = os.path.join("",model_dir)
model =CNN_bn(number_classes=10)
model.load_state_dict(torch.load(model_path))
all_preds_pytorch = []
all_labels_pytorch = []
all_probs_pytorch = []
model.eval()
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Step")):
#    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        probs = F.softmax(outputs, dim=1)[:, 1]  # probability for class 1
        all_probs_pytorch.extend(probs.cpu())
        all_preds_pytorch.extend(preds.cpu().numpy().flatten())
        all_labels_pytorch.extend(labels.numpy())

#----------------------------------------
# Métricas.
#----------------------------------------
agri_class_labels = ["non-agri", "agri"]
metrics_pytorch = model_metrics(all_labels_pytorch, all_preds_pytorch, all_probs_pytorch, agri_class_labels)
mertics_list = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
for k in mertics_list:
    print("{:<18} {:<15.4f}".format('\033[1m'+k+'\033[0m', metrics_pytorch[k]))
#----------------------------------------
# Gráficas.
#----------------------------------------
y_true = np.array(all_labels_pytorch)
y_pred = np.array(all_preds_pytorch)
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=agri_class_labels)
disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)  # Dibujar la matriz
plt.title("Matriz de Confusión")
plt.savefig("matriz_confusion.png", dpi=300, bbox_inches='tight')  #La guardamos como PNG

# Figura para Accuracy
epochs = range(1, len(loss_history['train']) + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs, acc_history['train'], label='Training Accuracy')
plt.plot(epochs, acc_history['val'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("Model Accuracy", dpi=300, bbox_inches='tight')  #La guardamos como PNG


# Figura para Loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, loss_history['train'], label='Training Loss')
plt.plot(epochs, loss_history['val'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig("Model Loss", dpi=300, bbox_inches='tight')  #La guardamos como PNG
