import torch
import numpy as np
import torch.nn.functional as F
from red_convolucional import CNN


def generar_entrega():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        datos = np.load('data/X_test.npz')
        X_test = datos['X']
    except Exception as e:
        print(f"Error al cargar X_test.npz: {e}")
        return

    modelo = CNN(dim_out=40)
    try:
        # Cargamos los pesos guardados por red_convolucional.py
        modelo.load_state_dict(torch.load('modelo_cnn.pth', map_location=device))
        modelo.to(device)
        modelo.eval()
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    probabilidades_totales = []

    print("Procesando imágenes de test...")
    with torch.no_grad():
        for i in range(len(X_test)):
            imagen = (X_test[i].astype(np.float32) - 136.03) / 59.55 # normalización
            tensor_img = torch.from_numpy(imagen).reshape(1, 1, 80, 70).to(device) # reshape

            # Obtener logits y aplicar Softmax para transformarlo en probabilidades
            logits = modelo(tensor_img)
            probabilidades = F.softmax(logits, dim=1)

            probabilidades_totales.append(probabilidades.cpu().numpy()[0])

    # Convertir la lista en una matriz de NumPy, forma (N_muestras, 40)
    Y = np.array(probabilidades_totales)

    print(f"Forma de la matriz Y: {Y.shape}")
    if Y.shape[1] != 40:
        print("El número de columnas no es 40.")
        return

    np.savez("Y_pred.npz", Y=Y)

    print("Fichero 'Y_pred.npz' generado con éxito.")


if __name__ == "__main__":
    generar_entrega()
