#For training model
import numpy, csv, math
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU, SReLU
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Adamax, Nadam
#from keras.utils import np_utils
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
#from keras.utils.visualize_util import plot
#import matplotlib.pyplot as plt
from ROOT import *

import theano
theano.config.gcc.cxxflags = '-march=corei7'
theano.config.openmp = False

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


print("******* Starting ********")
eos_path = "/eos/user/m/mmelodea/Keras/HiggsAnalysis/run2/KerasVBF/DNN_optimized_results/DataAugmentation/"
  
reader = numpy.loadtxt(eos_path+"FiltSamples/Higgs13TeV_train_118_130_ggh.csv", delimiter=",")
X_train = reader[:,0:18].astype(float) #using only pt, eta, phi
Y_train = reader[:,18]
#print X_train
print Y_train
nevents = len(Y_train)
print("Training from %i events..." % nevents)

reader2 = numpy.loadtxt(eos_path+"FiltSamples/Higgs13TeV_test_118_130_ggh.csv", delimiter=",")
X_validate = reader2[:,0:18].astype(float)
Y_validate = reader2[:,18]

    
# create model
model = Sequential()
  
model.add( Dense(60, init='uniform', input_shape=(18,), activation='relu') )
model.add( Dense(35, init='uniform', activation='relu') )
model.add( Dense(15, init='uniform', activation='relu') )
model.add( Dense(3, init='uniform', activation='relu') )
  
model.add( Dense(1, init='uniform', activation='sigmoid') )

#saves the model  
model_json = model.to_json()
smodel = "higgs_ggh_vbf_118-130_model.json"
with open(smodel,"w") as json_file:
  json_file.write(model_json)

  
#callbacks definitions
checkpoint1 = ModelCheckpoint("ggh_validation_best_acc_weights.h5",monitor='val_acc',verbose=0,save_best_only=True,mode='max')
checkpoint2 = ModelCheckpoint("ggh_validation_best_loss_weights.h5",monitor='val_loss',verbose=0,save_best_only=True,mode='min')
checkpoint3 = ModelCheckpoint("ggh_train_best_acc_weights.h5",monitor='acc',verbose=0,save_best_only=True,mode='max')
checkpoint4 = ModelCheckpoint("ggh_train_best_loss_weights.h5",monitor='loss',verbose=0,save_best_only=True,mode='min')
callbacks_list = [checkpoint1,checkpoint2,checkpoint3,checkpoint4]

  
# Optimizers
nepochs = 100

#learning_rate = 0.0001
#decay_rate = learning_rate/nepochs
#opt = SGD(learning_rate); #sthocastic gradient descent (best for the invariant mass and weights)
opt = Adam() #adaptive moment estimation (best known) (works fine for 3x, 3x2, sin(x)/x)
#opt = RMSprop()
#opt = Adagrad()
#opt = Adadelta()
#opt = Nadam()
#opt = Adamax()

#compiles the model
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

#accuracy function monitor
accuracy_monitor = EarlyStopping(monitor='loss', patience=0, verbose=0, mode='auto')

#adjusts model to data
model.fit(X_train, Y_train, validation_data=(X_validate,Y_validate), nb_epoch=nepochs, batch_size=37, callbacks=callbacks_list, verbose=2)
  

#saving weights
sweight = "higgs_ggh_vbf_118-130_weights.h5"
model.save_weights(sweight)


#final test
reader3 = numpy.loadtxt(eos_path+"FiltSamples/Higgs13TeV_ftest_118_130_ggh.csv", delimiter=",")
fX_test = reader3[:,0:18].astype(float)
fY_test = reader3[:,18]


model.load_weights(sweight)
scores = model.evaluate(fX_test, fY_test, verbose=0)
print("Accuracy for final weights: %.2f%%" % (scores[1]*100))

model.load_weights("ggh_validation_best_acc_weights.h5")
scores = model.evaluate(fX_test, fY_test, verbose=0)
print("Accuracy for best val_acc:  %.2f%%" % (scores[1]*100))

model.load_weights("ggh_validation_best_loss_weights.h5")
scores = model.evaluate(fX_test, fY_test, verbose=0)
print("Accuracy for best val_loss: %.2f%%" % (scores[1]*100))

model.load_weights("ggh_train_best_acc_weights.h5")
scores = model.evaluate(fX_test, fY_test, verbose=0)
print("Accuracy for best train_acc:  %.2f%%" % (scores[1]*100))

model.load_weights("ggh_train_best_loss_weights.h5")
scores = model.evaluate(fX_test, fY_test, verbose=0)
print("Accuracy for best train_loss: %.2f%%" % (scores[1]*100))
