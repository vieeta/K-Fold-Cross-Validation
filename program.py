import pandas as pd
import numpy as np
data = pd.read_csv("iris.csv")
data

#Search unique class inspired by www.geeksforgeeks.org
def unique(list):
    unique_list = []
    for x in list:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list
    
def datatodatalabel(data, name='name'):
  label = unique(data[name])
  for i in range(len(label)): 
    globals()['d_' + str(label[i])] = [data.loc[data['species'] == label[i]]]

def concat_n(data, name='name'):
  n = unique(data[name])
  for i in range(len(n)): 
    globals()['concat_' + str(n[i])] = pd.concat(globals()['d_'+str(n[i])], sort=False)

def splitdf_n(k,data, name='name'):
  n = unique(data[name])
  for i in range(len(n)): 
    globals()['fold_' + str(n[i])] = np.array_split(globals()['concat_' + str(n[i])], k)

def train_validationdata(k,data, name='name'):
  n = unique(data[name])
  com = [] 
  val_com = []
  for x in range(k):
    #copy data from fold (ex. fold_setosa to train_setosa consist of all fold (1 to 5))
    for i in range(len(n)): 
      globals()['train_' + str(n[i])] = globals()['fold_' + str(n[i])].copy()
    #validation consist of one fold (the k-fold)
    for i in range(len(n)): 
      globals()['validation_' + str(n[i])] = globals()['fold_' + str(n[i])][x]
    #delete the fold that are use in validation data, so training data now consist of k-1 fold
    for i in range(len(n)): 
      del globals()['train_' + str(n[i])][x]
    #concat the training data for each class 
    for i in range(len(n)): 
      globals()['train_'+str(n[i])] = pd.concat(globals()['train_'+str(n[i])], sort=False)
    #concat train_setosa, train_versicolor, train_virginica and then combine all to training data
    for i in range(len(n)): 
      com.append(globals()['train_'+str(n[i])])
    train = pd.concat(com, sort=False)
    #concat validation data(list -> dataframe)
    for i in range(len(n)): 
      val_com.append(globals()['validation_'+str(n[i])])
    validation = pd.concat(val_com, sort=False)
    print("K = ", x+1)
    print("train ", x+1)
    print(train)
    print("validation ", x+1)
    print(validation)

def stratifiedcrossfold(k, data, name='name'):
  datatodatalabel(data, name)
  concat_n(data, name)
  splitdf_n(k,data,name)
  train_validationdata(k, data, name)
  
stratifiedcrossfold(5, data, 'species')
