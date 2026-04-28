import torch.nn as nn
import numpy as np
import random
from Modulo1 import MiDataset
from Modulo1 import train
import torch

# TODO: comparar Adam y SGD en CNN (supuestamente funciona mejor Adam)
class CNN(nn.Module):
    def __init__(self, dim_out):
        super(CNN, self).__init__()
        '''
        1x70x80 -convolución-> 32x70x80 -relu-> 32x70x80 -pooling-> 32x35x40
        32x35x40 -convolución-> 64x35x40 -relu-> 64x35x40 -pooling-> 64x17x20
        64x17x20 -linealizamos-> 21760 -fc-> 40

        1x70x80 : dimensiones de la imagen original, entrada de la CNN
        40 : dimensiones del vector de salida (40 clases)
        '''
        # Entrada : 1x70x80
        self.convolution1 = nn.Conv2d(  # conv2d porque trabajamos con imágenes (alto*ancho)
            1,  # escala de grises -> un canal de entrada
            32,  # 32 filtros -> 32 canales de salida
            kernel_size=3,  # tamaño del filtro -> 3x3
            padding=1,
            # un aro de 0s alrededor de la iamgen (porque tenemos que poder situar el centro del filtro sobre cada pixel si no quremos encoger la imagen)
            stride=1  # un salto cada vez
        )  # Salida: 32x70x80
        self.bn1 = nn.BatchNorm2d(32) # NUEVO: añadido batch normalization a la versión anterior
        self.relu1 = nn.ReLU()  # Aplicamos la relu a cada valor del tensor: max(0, valor)
        self.pooling1 = nn.MaxPool2d(  # seleccionamos el valor más alto de cada zona 2x2
            kernel_size=2,
            stride=2,
            padding=0
        )  # Resultado: un tensor 32x35x40 (se reduce el tamaño a la mitad9

        self.convolution2 = nn.Conv2d(
            32,  # el número de canales de entrada es igual al número de canales de salida de la anterior
            64,  # Aplicamos 64 al tensor -> se amplian los canales a 64
            kernel_size=3,  # tamaño del filtro 3x3
            padding=1,  # una capa de relleno de 0s alrededor para no perder tamaño
            stride=1  # salto de 1
        )  # salida : tensor
        self.bn2 = nn.BatchNorm2d(64) # NUEVO
        self.relu2 = nn.ReLU()
        self.pooling2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            padding=0
        )  # salida : tensor 64x17x20

        # la salida se pasa por una fully connected, 64x17x20
        self.fc = nn.Linear(64*17*20, dim_out)

    def forward(self, x):
        # primera convolucion
        x = self.convolution1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pooling1(x)
        # segunda convolución
        x = self.convolution2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pooling2(x)
        # tranasformamos la matriz a un vector
        x = x.view(-1, 64*17*20)
        # la pasamos por la fully connected
        x = self.fc(x)
        return x

if __name__=="__main__":
    semillas = [42,  # semilla por defecto
                0, 1, 2, 15]  # otras semillas para determinar el mejor número de épocas
    semilla = semillas[4] # semilla que tiene mejor inicialización de acuerdo con los experimentos
    # TODO: probar con diferentes semillas para determinar el número de épocas óptimo
    np.random.seed(semilla) # mismo shuffling cada vez (el mezclado de datos será siempre el mismo, necesario para poder reproducir experimentos)
    torch.manual_seed(semilla) # resultados de los experimentos reproducibles. Garantiza que la inicialización de los pesos sea siempre la misma
    random.seed(semilla) # Volteo de imágnes con prabilidad de 0.5, (data augmentation de Modulo1) (de nuevo, para garantizar que los experimentos sean reproducibles)

    X = np.load('data/X_train.npz')['X']
    Y = np.load('data/Y_train.npz')['Y']
    '''
    Fundamental realizar una partición aleatoria para que haya una muestra representativa en el dataset de validación
    Primero se "baraja" el dataset con np.random.permutation para obligar a un orden aleatorio de las imágenes.
    Luego se realiza un corte en el dataset ya barajado al 0.8. Explicación visual:
     imágenes -> [0, 1, 2, 3, 4, 5]
     p -> [3, 2, 1, 0, 5, 4]
     la imágen en posición 0 pasa a la posición 3, la 1 a la 2 y así sucesivamente
     
    '''
    # p = np.random.permutation(len(X)) # creamos un vector con 240 índices (0-239) desordenados aleatoriamente
    # X, Y = X[p], Y[p] # las etiquetas seguirán correspondidendo a sus respectivas imágenes (no se moverán)
    # corte = int(len(X)*0.8) # Las primeras 192 imágenes se usarán para entrenamiento, el resgo para validación
    #
    #
    # dataset_entrenamiento = MiDataset(X[:corte], Y[:corte], aumentar=True)
    # dataset_validation = MiDataset(X[corte:], Y[corte:], aumentar=False)

    #____________________________________________________________________
    # --- Esta vez vamos a asignar 40 clases a validación (en lugar de 48 o el 20%)
    X_train_list, Y_train_list = [], []
    X_val_list, Y_val_list = [], []

    for clase in range(40):
        indices = np.where(Y == clase)[0]
        # barajamos para que la foto de validación no sea siempre la misma
        np.random.shuffle(indices)

        # seleccionamos la primera imagen para validación (ya están barajadas así que la posición no importa)
        idx_val = indices[0]
        X_val_list.append(X[idx_val])
        Y_val_list.append(Y[idx_val])

        # Las otras 5 imágenes las usamos para entrenamiento
        idx_train = indices[1:]
        X_train_list.append(X[idx_train])
        Y_train_list.append(Y[idx_train])

    # Convertimos las listas de nuevo a arrays de numpy
    X_train = np.concatenate(X_train_list, axis=0)
    Y_train = np.concatenate(Y_train_list, axis=0)
    X_val = np.array(X_val_list)
    Y_val = np.array(Y_val_list)

    # Creamos los datasets con los nuevos conjuntos
    dataset_entrenamiento = MiDataset(X_train, Y_train, aumentar=True)
    dataset_validation = MiDataset(X_val, Y_val, aumentar=False)
    #____________________________________________________________________
    device = "cuda" if torch.cuda.is_available() else "cpu"

    modelo = CNN(dim_out=40).to(device)

    historial = train(
        modelo=modelo,
        dataset_entrenamiento=dataset_entrenamiento,
        dataset_validation=dataset_validation,
        epocas=30, # de acuerdo con los experimentos realizados, el algoritmo aprende hasta la iteración número 15 y luego oscila
        batch_size=32,
        lr=0.001,
        device=device,
        nombre_modelo="modelo_cnn",
    )
