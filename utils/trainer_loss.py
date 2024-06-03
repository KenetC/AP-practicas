import numpy as np 
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def size_output(W,F,P,S): 
    return round((W-F+2*P )/ S) + 1 

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice
