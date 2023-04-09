import torch 
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

#We first get the dataset (using sklearn datasets)
#import sklearn
#from sklearn.datasets import fetch_openml


def logistic_reg_torch():
    #x.to_pickle("data.pkl"), data points are stored in this pickle file
    #y.to_pickle("label.pkl"), labels are stored in this pickle file
    
    #Data Loading etc..
    # open a file, where you stored the pickled data
    file = open('data.pkl', 'rb')
    # dump information to that file
    data = pickle.load(file)
    # close the file
    file.close()


    # open a file, where you stored the pickled data
    file = open('label.pkl', 'rb')
    # dump information to that file
    labels = pickle.load(file)
    # close the file
    file.close()
    #Now take any two classes (for two class classification)

    mask = (labels == '3') | (labels== '8')
    data_2=data[mask]
    labels_2=labels[mask]

    numpydata=data_2.to_numpy()
    numpylabel=labels_2.to_numpy()

    #split the data and labels into training and testing

    
    #plt.imsave('temp.png',numpydata_train[6,:].reshape(28,28))
    #Divide into test and train. Convert the numpy into tensor data 

    #Image data
    numpydata_train=numpydata[0:1000,:]            #First 10000 for training (for 2 classes)
    tdata_train=torch.from_numpy(numpydata_train)  #t stands for tensor
    numpydata_test=numpydata[10000:,:]
    tdata_test=torch.from_numpy(numpydata_test)
    

    #Label data 

    numpylabel_train=numpylabel[0:1000]
    tlabel_train=numpylabel_train
    trainlabeldata=torch.tensor(list(map(binariselabel,tlabel_train)))

    numpylabel_test=numpylabel[10000:]
    tlabel_test=numpylabel_test 
    testlabeldata=torch.tensor(list(map(binariselabel,tlabel_test)))
    groundtruthtest=testlabeldata.view(-1,1)  #converting into a row wise view 

    
    #iteration parameters
    batch_size = 50
    n_iters = 200
    num_epochs = n_iters / (1000 / batch_size)
    num_epochs = int(num_epochs)

    #Model Class

    class LogisticRegression(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(LogisticRegression, self).__init__()
            self.linear = torch.nn.Linear(input_dim, output_dim, dtype=float)
            
        def forward(self, x):
            outputs = (self.linear(x))
            return outputs
    
    input_dim = 28*28
    output_dim = 1
    model = LogisticRegression(input_dim, output_dim)

    criterion = nn.BCEWithLogitsLoss()
    learning_rate = 0.001
    #torch.optim.(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False,foreach=None, maximize=False, capturable=False, differentiable=False, fused=None)
    #label_df=one_hot()

    #Divide the training data into batches

    databatches = torch.chunk(tdata_train, batch_size, dim=0)
    labelbatches = torch.chunk(trainlabeldata, batch_size, dim=0)

    iter=0
    for epoch in range(num_epochs):

        for count, item in enumerate(databatches):
            image=item
            labeldata=labelbatches[count]
            #labeldata=torch.tensor(list(map(binariselabel,tlabel_train)))
            images = image.view(-1, 28*28).requires_grad_()
            
            # Select the row corresponding to labeldigit, and remove the first column
            #row = label_df.loc[label_df['labels'] == labeldigit].iloc[:, 1:]
            # Convert the values to a PyTorch tensor
            #labelonehot = torch.tensor(row.values).float() 
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Forward pass to get output/logits
            # # 100 x 10
            outputs = model.forward(images)
            # # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labeldata.view(-1,1).float())
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()
            iter += 1
            
            if iter % 10 == 0:
                testimages=tdata_test    
                testoutputs = model(testimages)
                m=nn.Sigmoid()
                outputprob = m(testoutputs)
                testoutputlabel = torch.where(outputprob>0.5,1,0)
                accuracy = 100-100*(sum(abs(groundtruthtest-testoutputlabel))/len(testoutputlabel))
                print('Iteration: {}. Loss: {}. Accuracy : {}'.format(iter, loss.item(),accuracy))
                

def one_hot():
    # creating initial dataframe
    labels = ('0','1','2','3','4','5','6','7','8','9')
    label_df = pd.DataFrame(labels, columns=['labels'])
    # generate binary values using get_dummies
    dum_df = pd.get_dummies(label_df, columns=["labels"], prefix=["Type_is"] )
    # merge with main df on key values
    label_df = label_df.join(dum_df)
    return label_df


def binariselabel(labeldata):
    if labeldata=='3':
        labelbin=0
    else:
        labelbin=1
    return labelbin
     




