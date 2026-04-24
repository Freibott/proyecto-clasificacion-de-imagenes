import torch
import numpy as np
import torch.nn.functional as F # para aplicar la softmax
from red_convolucional import CNN

# parte del código donde se realizarán las predicciones en el conjunto de datos nuevos
def test_modelo():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    datos = np.load('data/X_test.npz')
    X_test = datos['X']

    # provalos la convolucional que es el modelo más apropiado para imágenes (la fully connected es muy susceptible de tener overfitting aunque arroje un acc mayor
    modelo = CNN(dim_out=40)
    modelo.load_state_dict(torch.load('modelo_cnn.pth', map_location=device))
    modelo.to(device)
    modelo.eval()

    vector_probabilidades = []

    print("Comenzando evaluación del modelo: ")
    with torch.no_grad():
        for i in range(len(X_test)):
            imagen = (X_test[i].astype(np.float32) - 136.03) / 59.55 # normalización
            tensor_img = torch.from_numpy(imagen).reshape(1, 1, 80, 70).to(device) # ajuste de las dimensiones
            logits = modelo(tensor_img) # logits
            probabilidades = F.softmax(logits, dim=1)
            vector_probabilidades.append(probabilidades.cpu().numpy()[0])

        Y_predicha = np.array(vector_probabilidades)
        np.savez("outputs/Y_test.npz", Y=Y_predicha)

if __name__ == "__main__":
    test_modelo()