import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM,BatchNormalization
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import losses
# import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

df=pd.read_csv("voice.csv")
df.label=df.label.replace(to_replace=['male','female'],value=[0,1])

x=df.values
min_max_scaler=preprocessing.MinMaxScaler()
x_scaled=min_max_scaler.fit_transform(x)
df=pd.DataFrame(x_scaled)

df.columns=['meanfreq','sd','median','Q25','Q75','IQR','skew','kurt','sp.ent','sfm','mode','centroid','meanfun','minfun','maxfun','meandom','mindom','maxdom','dfrange','modindx','label']

attr=df[['meanfreq','sd','median','Q25','Q75','IQR','skew','kurt','sp.ent','sfm','mode','centroid','meanfun','minfun','maxfun','meandom','mindom','maxdom','dfrange','modindx']]
targets=df['label']
X_train,X_test,y_train,y_test=train_test_split(attr,targets,train_size=0.8,test_size=0.2,random_state=10)

# np.array(X_train)
# np.array(X_test).shape
# np.array(y_train).shape
# np.array(y_test).shape

def lossPerCostGraph(l1,l2,l3,l4,l5,l6,graphTitle):
    fontp=FontProperties()
    fontp.set_size('medium')
    p1,=plt.plot(l1,label='alpha=1.0')
    p2,=plt.plot(l2,label='alpha=0.5')
    p3,=plt.plot(l3,label='alpha=0.1')
    p4,=plt.plot(l4,label='alpha=0.01')
    p5,=plt.plot(l5,label='alpha=0.001')
    p6,=plt.plot(l6,label='alpha=0.0001')
    plt.legend(handles=[p1,p2,p3,p4,p5,p6],title='title',bbox_to_anchor=(1.05,1),loc='upper left',prop=fontp)
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.title(graphTitle)
    plt.show()

def accuracyAlphaGraph(accuracyList,alphaList,graphTitle):
    plt.plot(alphaList,accuracyList)
    plt.ylabel('accuracy')
    plt.xlabel('alpha')
    plt.title(graphTitle)
    plt.show()
    
def seqModel(alpha):
    model=Sequential()
    model.add(Dense(2,activation='sigmoid',input_shape=(20,)))
    optVal=Adam(lr=alpha,decay=1e-6)
    lossVal=losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optVal,loss=lossVal,metrics=['accuracy'])
    resultVal=model.fit(np.array(X_train),np.array(y_train),epochs=1000)
#     lossPerCostGraph(resultVal.history['loss'],'Sequential model cost per epoch Graph')
    test_ds_loss,test_ds_accuracy=model.evaluate(np.array(X_test),np.array(y_test),verbose=2)
    return resultVal.history['loss'],test_ds_accuracy
    
def twoLayerModel(alpha):
    model=Sequential()
    model.add(Dense(10,activation='sigmoid',input_shape=(20,),name='Hidden_Layer'))
    model.add(Dense(4,activation='sigmoid',input_shape=(10,),name='Output_Layer'))
    optVal=Adam(lr=alpha,decay=1e-6)
    lossVal=losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optVal,loss=lossVal,metrics=['accuracy'])
    resultVal=model.fit(np.array(X_train),np.array(y_train),epochs=1000)
#     lossPerCostGraph(resultVal.history['loss'],'Two Layer model cost per epoch Graph')
    test_ds_loss,test_ds_accuracy=model.evaluate(np.array(X_test),np.array(y_test),verbose=2)
    return resultVal.history['loss'],test_ds_accuracy



#seqModel
list1,accuracy1=seqModel(1.0)
list2,accuracy2=seqModel(0.5)
list3,accuracy3=seqModel(0.1)
list4,accuracy4=seqModel(0.01)
list5,accuracy5=seqModel(0.001)
list6,accuracy6=seqModel(0.0001)
lossPerCostGraph(list1,list2,list3,list4,list5,list6,'sequential model loss/cost graph')
accuracyList=[accuracy1,accuracy2,accuracy3,accuracy4,accuracy5,accuracy6]
alphaList=[1.0,0.5,0.1,0.01,0.001,0.0001]
accuracyAlphaGraph(accuracyList,alphaList,'Accuracy vs Alpha in sequential model')
    

# twoLayerModel for activation function(sigmoid,sigmoid)
l1,accuracy1=twoLayerModel(1.0,'sigmoid','sigmoid')
l2,accuracy2=twoLayerModel(0.5,'sigmoid','sigmoid')
l3,accuracy3=twoLayerModel(0.1,'sigmoid','sigmoid')
l4,accuracy4=twoLayerModel(0.01,'sigmoid','sigmoid')
l5,accuracy5=twoLayerModel(0.001,'sigmoid','sigmoid')
l6,accuracy6=twoLayerModel(0.0001,'sigmoid','sigmoid')
lossPerCostGraph(l1,l2,l3,l4,l5,l6,'two layer model loss/cost graph')
accuracyList=[accuracy1,accuracy2,accuracy3,accuracy4,accuracy5,accuracy6]
alphaList=[1.0,0.5,0.1,0.01,0.001,0.0001]
accuracyAlphaGraph(accuracyList,alphaList,'Accuracy vs Alpha in two layer model')

# twoLayerModel for activation functions (relu,relu)
l1,accuracy1=twoLayerModel(1.0,'relu','relu')
l2,accuracy2=twoLayerModel(0.5,'relu','relu')
l3,accuracy3=twoLayerModel(0.1,'relu','relu')
l4,accuracy4=twoLayerModel(0.01,'relu','relu')
l5,accuracy5=twoLayerModel(0.001,'relu','relu')
l6,accuracy6=twoLayerModel(0.0001,'relu','relu')
lossPerCostGraph(l1,l2,l3,l4,l5,l6,'two layer model loss/cost graph')
accuracyList=[accuracy1,accuracy2,accuracy3,accuracy4,accuracy5,accuracy6]
alphaList=[1.0,0.5,0.1,0.01,0.001,0.0001]
accuracyAlphaGraph(accuracyList,alphaList,'Accuracy vs Alpha in two layer model')


    
# twoLayerModel for activation functions (tanh,sigmoid)
l1,accuracy1=twoLayerModel(1.0,'tanh','sigmoid')
l2,accuracy2=twoLayerModel(0.5,'tanh','sigmoid')
l3,accuracy3=twoLayerModel(0.1,'tanh','sigmoid')
l4,accuracy4=twoLayerModel(0.01,'tanh','sigmoid')
l5,accuracy5=twoLayerModel(0.001,'tanh','sigmoid')
l6,accuracy6=twoLayerModel(0.0001,'tanh','sigmoid')
lossPerCostGraph(l1,l2,l3,l4,l5,l6,'two layer model loss/cost graph')
accuracyList=[accuracy1,accuracy2,accuracy3,accuracy4,accuracy5,accuracy6]
alphaList=[1.0,0.5,0.1,0.01,0.001,0.0001]
accuracyAlphaGraph(accuracyList,alphaList,'Accuracy vs Alpha in two layer model')



# twoLayerModel for activation functions (relu,sigmoid)
l1,accuracy1=twoLayerModel(1.0,'relu','sigmoid')
l2,accuracy2=twoLayerModel(0.5,'relu','sigmoid')
l3,accuracy3=twoLayerModel(0.1,'relu','sigmoid')
l4,accuracy4=twoLayerModel(0.01,'relu','sigmoid')
l5,accuracy5=twoLayerModel(0.001,'relu','sigmoid')
l6,accuracy6=twoLayerModel(0.0001,'relu','sigmoid')
lossPerCostGraph(l1,l2,l3,l4,l5,l6,'two layer model loss/cost graph')
accuracyList=[accuracy1,accuracy2,accuracy3,accuracy4,accuracy5,accuracy6]
alphaList=[1.0,0.5,0.1,0.01,0.001,0.0001]
accuracyAlphaGraph(accuracyList,alphaList,'Accuracy vs Alpha in two layer model')


    







