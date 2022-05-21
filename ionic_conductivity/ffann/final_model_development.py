
###FFANN model development for Fluid Phase ML paper############


######################loading packages###########


import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.wrappers.scikit_learn import KerasRegressor
from keras.activations import *
from keras.optimizers import Adam
import pickle
import tensorflow as tf

seed =7
ct=0


plt.rcParams["figure.figsize"] = [12,10]
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "medium"
plt.rcParams["text.color"] = "darkred"
plt.rcParams['axes.linewidth'] = 2.2
#plt.rcParams['xtick.major.size'] = 20
#plt.rcParams['xtick.major.width'] = 4
#plt.rcParams['xtick.minor.size'] = 10
#plt.rcParams['xtick.minor.width'] = 2
#plt.rcParams['ytick.major.size'] = 20
#plt.rcParams['ytick.major.width'] = 4
#plt.rcParams['ytick.minor.size'] = 10
#plt.rcParams['ytick.minor.width'] = 2

#######font parameters for plot#################

font = {'family': 'STIXGeneral',
        'color':  'darkred',
        'weight': 'bold',
        'size': 12,
        }


##################reading the training data#####################


dn = pd.read_csv('final_converted2.txt',sep="\t",index_col=None,header=None)
dn=dn.rename(columns=dn.iloc[0],copy=False).iloc[1:].reset_index(drop=True) 
#df= df.set_index([df.columns[0],df.columns[1]])

print(dn.columns)
print(dn.info())


########################### data processing #########################################

df = dn.drop(['Cation_Name', 'Cation_smiles', 'Anion_Name', 'Anion_smiles'],axis=1)
print(df.columns)
print(df.info())
#dh = dn[['Cation_smiles','Anion_smiles']]

###this will take the cation and anion smiles only

dname = dn.iloc[:,0:4]
print(dname.columns)


df=df.apply(pd.to_numeric)


###################take only the cation descriptors

dcation = df.iloc[:,0:196]
print(dcation.columns)


####################drop any columns with NA

dcation =dcation.dropna(axis='columns') 

p= (dcation.columns[dcation.sum() == 0 ])   ##remove any columns that have all values of zero

dcation =dcation.drop(columns=p)  


############################feature correlation################################

#########cation feature correlation###################################

corr = dcation.corr()
print(corr.info())


columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= abs(0.9):
            if columns[j]:
               columns[j] = False
selected_columns = dcation.columns[columns]
print(selected_columns)
dcation = dcation[selected_columns]

print(dcation.info())


cation_column = dcation.columns.tolist()
print((cation_column))


####anion feature selection

danion = df.iloc[:,196:392]
print(danion.columns)

danion =danion.dropna(axis='columns') 

p= (danion.columns[danion.sum() == 0 ])  
danion =danion.drop(columns=p)  


####correlation feature for anion##############

corr = danion.corr()
print(corr.info())


columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= abs(0.9):
            if columns[j]:
               columns[j] = False
selected_columns = danion.columns[columns]
print(selected_columns)
danion = danion[selected_columns]

print(danion.info())

anion_column = danion.columns.tolist()
print((anion_column))


#####save the column name#######

np.savetxt('cation_column.txt', cation_column, delimiter=";", fmt="%s")
np.savetxt('anion_column.txt', anion_column, delimiter=";", fmt="%s")



################processing property################

dp = df[['Delta(prev)','Pressure/kPa','Temperature/K','Electrical_conductivity[Liquid]/S/m']]

do = pd.concat([dname,dcation,danion,dp],axis=1,sort=False)
print(do.info())

do =do.dropna(axis='rows')
print(do.info())



###combine the name, cation , anion and properties


print(do['Pressure/kPa'].describe())


####removing duplicate data###########

do= do.sort_values('Delta(prev)', ascending=True).drop_duplicates(['Cation_smiles','Anion_smiles','Temperature/K','Pressure/kPa'],keep='last').sort_index()


###drop the delta error column as it is no longer required#######


do = do.drop(['Delta(prev)'],axis=1)

do = do.reset_index(drop=True)



#####filter property##############


do = do[do['Pressure/kPa'] == 101][do["Temperature/K"] < 475][do["Temperature/K"] > 275][
        do['Electrical_conductivity[Liquid]/S/m']>0]



len_row=do.shape[0]

len_coul=do.shape[1]



####take all the property after the 4th column as the first four column contains name, smiles for cation and anion
###ignore the last column as it is the property of interest########################

X= do.iloc[:,4:len_coul-1].values  



y = do.iloc[:,len_coul-1].values







##########################################data normalization


########data normalization
sc =MinMaxScaler()
sy = MinMaxScaler()
X = sc.fit_transform(X)

y = np.log10(y)
y= y.reshape(-1,1)
y = sy.fit_transform(y)
#########################################################################################

joblib.dump(sc, 'scaler.gz') 
joblib.dump(sy, 'scaler2.gz') 

########################################################################################
#########divide the data into training/test set#####################################

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.1,random_state=50)


##############neural network block#######################################

opt= Adam(learning_rate=0.002)
regressor = Sequential()
regressor.add(Dense(units=1*(len_coul-5),activation='relu',input_dim=len_coul-5)) ###first hidden layer
regressor.add(Dense(units=24,activation='relu'))  ## second hidden layer
regressor.add(Dense(units=1,activation='relu'))   ###final layer
regressor.compile(optimizer= opt,loss='mse',metrics=['mae','acc','mse'])
history=regressor.fit(X_train,y_train,batch_size =100,epochs=450)

########################################################################################

#################predict train data#############################################

y_tra = regressor.predict(X_train)
y_ptrain=y_tra.reshape(-1,1)
ys=y_train.reshape(-1,1)
ypn  = sy.inverse_transform(y_ptrain) 
ytn  = sy.inverse_transform(ys) 
ypn = 10**(ypn)     ####model
ytn = 10**(ytn)      ####experiment
################################################################

print('r2 of train set is',r2_score(ytn,ypn))
print('mae of train set is',mean_absolute_error(ytn,ypn))
print('rmsd of train set is',mean_squared_error(ytn,ypn,squared=False))
#####################################################################


pyplot.scatter(ytn,ypn,label='Training Set')
plt.xlabel('Experiment Ionic Conductivity S/m',fontdict=font)
plt.ylabel('Predicted Ionic Conductivity S/m',fontdict=font)
pyplot.legend()



############################################save data##############################

ypn = np.concatenate(ypn,axis=0)
ytn = np.concatenate(ytn,axis=0)
ypn = ypn.tolist()
ytn = ytn.tolist()
####################################################################################
make = np.column_stack((ypn,ytn))
df = pd.DataFrame(make,columns=['Model_S/m','Exp_S/m'])
df.to_csv('nn_model_train_set.csv',sep=';',index=False) 



#########predict test data####################################


y_tra = regressor.predict(X_test)
y_ptrain=y_tra.reshape(-1,1)
ys=y_test.reshape(-1,1)
ypn  = sy.inverse_transform(y_ptrain) 
ytn  = sy.inverse_transform(ys) 
ypn = 10**(ypn)     ####model
ytn = 10**(ytn)      ####experiment


print('r2 of test set is',r2_score(ytn,ypn))
print('mae of test set is',mean_absolute_error(ytn,ypn))
print('rmsd of test set is',mean_squared_error(ytn,ypn,squared=False))

pyplot.scatter(ytn,ypn,label='Test Set')
plt.xlabel('Experiment Ionic Conductivity S/m',fontdict=font)
plt.ylabel('Predicted Ionic Conductivity S/m',fontdict=font)
pyplot.legend()
plt.savefig('train.png') 



######################save test data#############################################


ypn = np.concatenate(ypn,axis=0)
ytn = np.concatenate(ytn,axis=0)
ypn = ypn.tolist()
ytn = ytn.tolist()
####################################################################################
make = np.column_stack((ypn,ytn))
df = pd.DataFrame(make,columns=['Model_S/m','Exp_S/m'])
df.to_csv('nn_model_test_set.csv',sep=';',index=False) 


fig, ax = plt.subplots()

plt.plot(history.history['mse'])  
plt.plot(history.history['mae'])
plt.savefig('history.png')

######save the model#############################################
  
regressor.save('testlog_model.h5')

