from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay
from sklearn.linear_model import LinearRegression
import torch
from torch import nn 
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np 
import skimage.io as io
import os 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Practica 3 
def plotScatter(x_data, y_data, title, fit_line=True):
    '''
    Definimos un método para mostrar las predicciones como un scatter plot y graficamos la recta de regresión para esos datos.
    '''
    plt.figure()
    plt.plot(x_data, y_data, '+')
    plt.xlabel('Valor real')
    plt.ylabel('Predicción')
    plt.title(title)

    if fit_line:
        X, Y = x_data.reshape(-1,1), y_data.reshape(-1,1)
        plt.plot( X, LinearRegression().fit(X, Y).predict(X) )

def graficos(model,X,y,Neuronas): 
    '''
    Muestro la lista que contiene los valores de la función de pérdida y una versión suavizada (rojo) para observar la tendencia
    '''
    print('Current device is', device)
    X_train, X_test = X
    y_train, y_test = y 
    # Dibujamos el ground truth vs las predicciones en los datos de entrenamiento
    py = model(torch.FloatTensor(X_train).to(device))
    y_pred_train = py.cpu().detach().numpy()
    plotScatter(y_train, y_pred_train, f"Training data de {Neuronas} Neuronas")

    # Dibujamos el ground truth vs las predicciones en los datos de test
    py = model(torch.FloatTensor(X_test).to(device))
    y_pred_test = py.cpu().detach().numpy()
    plotScatter(y_test, y_pred_test, f"Test data de {Neuronas} Neuronas")

    print ("MSE medio en training: " + str(((y_train - y_pred_train)**2).mean()))
    print ("MSE medio en test: " + str(((y_test - y_pred_test)**2).mean()))

def showTestResults(model,data_test):
    '''
        Plot de la matriz de confuncion para un modelo dado  
    '''
    plt.rcParams['figure.figsize'] = [12, 10]
    
    # Genero un data loader para leer los datos de test 
    loader_test = DataLoader(dataset=data_test, batch_size=10000, shuffle=False)
    x_test = list(loader_test)[0][0]
    y_test = list(loader_test)[0][1]

    x_test = x_test.to(device)

    pred = model(x_test)

    # Extraigo el índice de la predicción con mayor valor para decidir la clase asignada
    pred_y = torch.max(pred.to("cpu"), 1)[1].data.numpy()

    # Imprimo el reporte de clasificación (accuracy, etc)
    print(classification_report(y_test, pred_y))

    # Computo la matriz de confusión y la muestro
    conf_mat = confusion_matrix(y_test, pred_y)
    plt.matshow(conf_mat, cmap='jet')

    for (i, j), z in np.ndenumerate(conf_mat):
        plt.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

    plt.title("Confusion matrix")
    plt.colorbar()

# Practica 4 

def visualizarImagenTest(imgNbr,args):
  # Leer el ground truth
  gt = io.imread(os.path.join(args['TEST_LABELS_DIR'], str(imgNbr) + ".png"),as_gray = True)
  gt = gt[:,:].astype(bool).astype(float)

  # Leer la imagen original
  img = io.imread(os.path.join(args['TEST_IMAGES_DIR'], str(imgNbr) + ".png"),as_gray = True)

  # Visualizar la imagen y el ground truth
  plt.figure(figsize = (9, 3), dpi = 100)

  plt.subplot(1,3,1)
  plt.grid(False)
  plt.title("Imagen de test: " + str(imgNbr))

  plt.imshow(img, cmap='gray')

  plt.subplot(1,3,2)
  plt.axis('off')
  plt.title("GT")

  plt.imshow(gt, cmap = 'viridis')

  plt.subplot(1,3,3)
  plt.axis('off')

  plt.title("GT superpuesto")
  plt.imshow(img, cmap='gray')
  plt.imshow(gt, cmap='jet', alpha = 0.25)

def g_loss_dice(history,BN): 
    wo = "con" if BN else "sin"
    plt.figure()
    plt.title(f"Loss vs epochs, {wo} BN")
    plt.plot(history['Train_Loss'])
    plt.legend(['Training Loss'])

    plt.figure()
    plt.title(f"Dice vs epochs, {wo} BN")
    plt.plot(history['Val_Dice'])
    plt.legend(['Validation Dice'])

def img_test(image,label,seg,BN:bool):
    binary = seg > 0.5
    wo = "con" if BN else "sin"

    plt.figure(dpi=100)

    plt.subplot(1,3,1)
    plt.axis('off')
    plt.imshow(image[0,:,:])
    plt.title('Test sample')

    plt.subplot(1,3,2)
    plt.axis('off')
    plt.imshow(label)
    plt.title('Label')

    plt.subplot(1,3,3)
    plt.axis('off')
    plt.imshow(binary.cpu().numpy())
    plt.title(f'Output segmentation, {wo} BN')

def Boxplots(grap_dice,grap_h,BN:bool):
    wo = "con" if BN else "sin" 
    plt.boxplot(grap_dice)
    plt.title(f'Boxplot Dice, {wo} BN')
    plt.ylabel('Valor')
    plt.show()

    plt.boxplot(grap_h)
    plt.title(f'Boxplot Haussdorf, {wo} BN')
    plt.ylabel('Valor')
    plt.show()