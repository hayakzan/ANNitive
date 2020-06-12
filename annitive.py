# ANNitive--ANN-based additive synthesis

# using a regression model for generating a new spectrum
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy import array
from numpy import savetxt
from numpy import genfromtxt
np.set_printoptions(suppress=True) # suppress scientific notation

# load the dataset
dataset_1 = genfromtxt('train_spectral_data_1.txt', delimiter=',')
dataset_2 = genfromtxt('train_spectral_data_2.txt', delimiter=',')
test_dataset = genfromtxt('test_spectral_data.txt', delimiter=',')

# split into input (X) and output (y) variables
X = dataset_1
X = X[~np.isnan(X)]
X = np.reshape(X, (-1, 2)) #transform to 2D
y = dataset_2
y = y[~np.isnan(y)]
y = np.reshape(y, (-1, 2))
Xtest = dataset_1
Xtest = Xtest[~np.isnan(Xtest)]
Xtest = np.reshape(Xtest, (-1, 2))

# define and fit the final model.
model = Sequential()
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='linear')) #due to two dimensions of the output, output layer should be 2
model.compile(loss='mse', optimizer='adam')
model.fit(X, y, epochs=5000, verbose=1)

# new instance
Xnew = Xtest
# make a prediction
ynew = model.predict(Xnew)

# inputs and predicted outputs
freqs = ynew[:,0:1]
amps = ynew[:,1:2]
amps = np.reshape(amps, (len(amps)))
newamps = []
newfreqs = []
# get rid of the negative amps
for i in range(len(amps)):
    if amps[i] > 0:
        newamps = np.append(newamps, amps[i])
        newfreqs = np.append(newfreqs, freqs[i])

# save for SC
np.savetxt('newFreqs.txt', [newfreqs], delimiter=',', header='[', footer=']', comments = '', fmt='%f')
np.savetxt('newAmps.txt', [newamps], delimiter=',', header='[', footer=']', comments = '', fmt='%f')
