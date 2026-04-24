import numpy as np

# Cargar datos
X = np.load('data/X_train.npz')['X']
Y = np.load('data/Y_train.npz')['Y']

clases, tam_muestra = np.unique(Y, return_counts=True)
print(X.shape)
print(X.dtype)
print(Y.shape)
print(Y.dtype)
print(f"Número de clases: {len(clases)}")
print(f" Número de muestras por clase: {tam_muestra}")

'''
(240, 5600) -> 240 imágenes, 5600 píxeles
uint8  -> debería ser float32
(240,) # contiene las etiquetas para cada una de las 240 imágenes
int64  -> debería ser float32
Número de clases: 40
 Número de muestras por clase: [6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
 6 6 6] -> dataset equilibrado
'''

X_float = X.astype(np.float32)
Y_float = Y.astype(np.float32)

media_X = X_float.mean()
std_X = X_float.std()

print(f'Media: {media_X}')
print(f'Desviación estándar: {std_X}')

'''
Media: 136.0331573486328
Desviación estándar: 59.553890228271484
'''