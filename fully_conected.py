import torch.nn as nn
import numpy as np
import random
from Modulo1 import MiDataset
from Modulo1 import train

import torch

'''
 La dimensión de entrada para la red fully connected son 70x80 = 5600, la dimensión de salida será un vector de
 probabildiades de 40 dimensiones (uno para cada clase).
 '''
class FCDNN(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        '''
        La FCDNN recibe un vecctor de entrada a diferencia de la CNN que recibe un tensor

        5.600 -producto matriz> 512 -batch normalization-> 512
        512 -producto matriz-> 216 -batchnormalization-> 256
        216 -produto matriz -> 40 (dim out) -relu> 40
        '''
        self.capa1 = nn.Linear(dim_in, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.capa2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.capa_salida = nn.Linear(256, dim_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        La función forward espera recibir un vector plano. Como al cargar los datos en el Modulo 1 le damos una forma
        de tensor (2 dimensiones), debemos volver a aplanarlo -> x = x.view(x.size(0), -1)
        :param x: vector plano
        :return: logits
        '''
        x = x.view(x.size(0), -1) # aplanamos la matriz para obtener el vector
        x = self.capa1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.capa2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.capa_salida(x)
        return x


if __name__ == "__main__":
    semillas = [42,  # semilla por defecto
                0, 1, 2]  # otras semillas para determinar el mejor número de épocas
    semilla = semillas[0]
    # Fijamos una semilla para conseguir siempres resultados iguales en los tests
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    random.seed(semilla) # necesaria para el data augmentation estocástico
    # cargamos los datos de entranamiento y las etiquetas
    X_datos = np.load('data/X_train.npz')["X"]
    Y_datos = np.load('data/Y_train.npz')["Y"]

    # dividimos el dataset: 80% train, 20% validación
    p = np.random.permutation(len(X_datos))# creación del vector con los valores 0 a 239 desordenados
    X_mezclado = X_datos[p] # asignamos cada par de valores (caras y etiquetas) a cada valor del subíndice.
    Y_mezclado = Y_datos[p]
    punto_corte = int(len(X_mezclado)*0.8) # las primeras 192 imágenes las usamos para entrenamiento el resto para validación

    dataset_train = MiDataset(X_mezclado[:punto_corte], Y_mezclado[:punto_corte], aumentar=True)
    dataset_validacion = MiDataset(X_mezclado[punto_corte:], Y_mezclado[punto_corte:], aumentar=False) # solo queremos validar las imágenes reales

    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = FCDNN(dim_in=5600, dim_out=40)



    print(f"Ejecutando test")
    historial = train(
        modelo=modelo,
        dataset_entrenamiento=dataset_train,
        dataset_validation=dataset_validacion,
        epocas=15, # hemos visto que el aprendizaje se satura a partir de la época 15 con diferentes semillas
        batch_size=32, # número de imágenes que procesa el modelo antes de actualizar pesos
        lr=0.01,
        device=device,
        nombre_modelo="modelo_fcdnn")
