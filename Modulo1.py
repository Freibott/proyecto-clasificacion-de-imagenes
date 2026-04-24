import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF



class MiDataset(Dataset):
    RANGO_ROTACION = 20  # ángulo máximo que se puede rstar o sumar a la orientación de la imagen

    def __init__(self, X, Y, aumentar=False):
        self.X = (X.astype(np.float32) - 136.03) / 59.55
        self.Y = Y
        self.aumentar = aumentar

    def __len__(self):
        return len(self.X)
    '''
    Cambios con respecto a la versión anterior: añadido data augmentation
    Por cada época se ven las 192 imágenes (el 80% del dataset), pero cada vez aplicamos una transformación aleatorio.
    Podríamos elegir un número de épocas teóricamente infinito
    En la práctica, llega un punto donde el modelo deja de aprender. 
    
    En el código, hemos establecido el número de épocas en base al punto en el quealcanza su punto de saturación. Por ejemplo en FCDNN:
    30 épocas * 192 imágenes/época = 5760 imágenes
    En la gráfica de la precisión se ve cómo al llegar a cierto valor añadir épocas deja de aumentar la precisión 
    (simplemente se queda oscilando)
    '''
    def __getitem__(self, i):
        imagen = torch.from_numpy(self.X[i].reshape(1, 80, 70)) # las imágenes tienen un alto de 80 y un ancho de 70
        etiqueta = torch.tensor(self.Y[i], dtype=torch.long)

        if self.aumentar:
            if random.random() < 0.5: # aplicamos el espejo de la imagen con una probabilidad del 50%
                imagen = TF.hflip(imagen)
            angulo = random.uniform(-self.RANGO_ROTACION, self.RANGO_ROTACION) # elegimos un ángulo al azar entre -20º y +20º
            imagen = TF.rotate(imagen, angulo) # una vez determinado el ángulo, rotamos la imagen con la función de torchvision

        return imagen, etiqueta

def calcular_precision(modelo, dataloader, device):
    modelo.eval()
    aciertos = 0
    total = 0
    with torch.no_grad():
        for imagenes, etiquetas in dataloader:
            imagenes = imagenes.to(device)
            etiquetas = etiquetas.to(device)
            salidas = modelo(imagenes) # predicciones
            _, predicciones = torch.max(salidas, 1)
            total += etiquetas.size(0)
            aciertos += (predicciones == etiquetas).sum().item()
    return (aciertos/ total)*100

def train(modelo, dataset_entrenamiento, dataset_validation, epocas, batch_size, lr, device, nombre_modelo="modelo_guardado", optimizador=None):
    loader_entrenamiento = DataLoader(dataset_entrenamiento, batch_size=batch_size,
                                      shuffle=True) # shuffle para que en cada época donde se recorren los datos se realice en un orden diferente
    loader_validation = DataLoader(dataset_validation, batch_size=batch_size, shuffle=False)
    error = nn.CrossEntropyLoss()
    if optimizador is None: # el optimizador por defecto es SGD+Momentum
        optimizador = torch.optim.SGD(modelo.parameters(), lr=lr, momentum=0.9)
    modelo.to(device)

    print("epoca;perdida;precision_train;precision_val;metrica")
    for epoca in range(epocas):
        modelo.train()
        perdida_acumulada = 0
        for imagenes, etiquetas in loader_entrenamiento:
            imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
            optimizador.zero_grad()
            predicciones = modelo(imagenes)
            perdida = error(predicciones, etiquetas)
            perdida.backward()
            optimizador.step()
            perdida_acumulada += perdida.item()


        precision_train = calcular_precision(modelo, loader_entrenamiento, device)/100
        precision_val = calcular_precision(modelo, loader_validation, device)/100
        # TODO: corregir la métrica
        _, _, metrica = evaluar_modelo(modelo, loader_validation, device)

        perdida_media = perdida_acumulada / len(loader_entrenamiento)

        print(f"{epoca};{perdida_media:.4f};{precision_train:.4f};{precision_val:.4f};{metrica:.4f}".replace('.', ','))


    torch.save(modelo.state_dict(), f'{nombre_modelo}.pth')
    print(f"Modelo guardado en {nombre_modelo}.pth")

def evaluar_modelo(modelo, loader, device):
    modelo.eval()
    total = 0
    aciertos_top1 = 0
    aciertos_top3 = 0

    with torch.no_grad():
        for imagenes, etiquetas in loader:
            imagenes, etiquetas = imagenes.to(device), etiquetas.to(device)
            salidas = modelo(imagenes)
            _, top3_indices = torch.topk(salidas, 3, dim=1)
            total += etiquetas.size(0)
            for i in range(etiquetas.size(0)):
                etiqueta = etiquetas[i]
                predicciones = top3_indices[i]
                if etiqueta == predicciones[0]:
                    aciertos_top1 += 1
                if etiqueta in predicciones:
                    aciertos_top3 += 1
    precision = aciertos_top1 / total
    top3 = aciertos_top3 / total
    puntuacion_final = (0.7*precision) + (0.3*top3)
    return precision, top3, puntuacion_final
