
def showTestResults(model):
    '''
        Plot de la matriz de confuncion para un modelo dado  
    '''
    plt.rcParams['figure.figsize'] = [12, 10]
    from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay

    # Genero un data loader para leer los datos de test 
    loader_test = DataLoader(dataset=mnist_test, batch_size=10000, shuffle=False)
    x_test = list(loader_test)[0][0]
    y_test = list(loader_test)[0][1]

    # Muevo los tensores a la GPU
    x_test = x_test.to(device)

    # Realizo las predicciones del modelo
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


def trainer(model, criterion, loader,optimizer,args, seed=123):
    '''
    training the model with the metrics and training and validation data. 
    '''
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Current device is', device)
    model = model.to(device)
    criterion = criterion().to(device)
    loss_list = []
    for i in range(args['num_epochs']):
        for x, y in loader:
            optimizer.zero_grad()
            #x = x.to(device).view(-1,input_features)
            x = x.to(device)
            y = y.to(device)
            # Realizo la pasada forward por la red
            loss = criterion(net(x), y)
            # Realizo la pasada backward por la red        
            loss.backward()
            # Actualizo los pesos de la red con el optimizador
            optimizer.step()
            loss_list.append(loss.data.item())
        # Muestro el valor de la función de pérdida cada 100 iteraciones        
        #if i > 0 and i % 100 == 0:
        print('Epoch %d, loss = %g' % (i, loss))
    plt.figure()
    loss_np_array = np.array(loss_list)
    plt.plot(loss_np_array, alpha = 0.3)
    N = 60
    running_avg_loss = np.convolve(loss_np_array, np.ones((N,))/N, mode='valid')
    plt.plot(running_avg_loss, color='red')
    plt.title("Función de pérdida durante el entrenamiento")

