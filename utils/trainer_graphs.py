import torch
from torch import nn 
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.linear_model import LinearRegression

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
def h(): 
    print("asdgfasd")
    
def trainer(model, criterion, args):
    '''
    training the model with training  dataset.
    suponemos que X_train y y_train estan ordenados de forma adecuada para la posterior comparacion de model(X_train) == y_train.
    '''
    # config GPU 
    seed = 123
    torch.manual_seed(seed)
    print('Current device is', device) 

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

    criterion = criterion.to(device)

    if args.get('dataset') is None: 
        X_train, y_train = args['X_train'],args['y_train']
        dataset = TensorDataset(torch.from_numpy(X_train).clone(), torch.from_numpy(y_train).clone())
    else:
        dataset = args['dataset']
    
    loader = DataLoader(dataset=dataset,batch_size=args['batch_size'],shuffle=True)

    loss_list = []
    for i in range(args['num_epochs']):
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            loss.backward()
            # Actualizo los pesos de la red con el optimizador
            optimizer.step()
            # Me guardo el valor actual de la función de pérdida para luego graficarlo
            loss_list.append(loss.data.item())
            # loss del minibatch // .item() valor numerico de la perdida.
            total_loss += loss.item() * y.size(0)
        total_loss/= len(loader.dataset)
        # Muestro el valor de la función de pérdida cada 100 iteraciones        
        if i > 0 and i % 1000 == 0:
            print('Epoch %d, loss = %g' % (i, total_loss))
    plt.figure()
    loss_np_array = np.array(loss_list)
    plt.plot(loss_np_array, alpha = 0.3)
    N = 60
    running_avg_loss = np.convolve(loss_np_array, np.ones((N,))/N, mode='valid')
    plt.plot(running_avg_loss, color='red')
    if args.get('Neuronas') is not None: 
        plt.title(f"Función de pérdida durante el entrenamiento para {args['Neuronas']} de neuronas")
    else:
        plt.title("Función de pérdida durante el entrenamiento")
    print(min(loss_np_array))

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

from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay
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
