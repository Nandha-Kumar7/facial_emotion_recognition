import sys, os
import pandas 
import numpy 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils




df=pandas.read_csv('Emotions_DATA.csv')





Xtrain,trainy,Xtest,testy=[],[],[],[]

for index, row in df.iterrows():
    val=row['_PIXELS_'].split(" ")
    try:
        if '_trainings_' in row['_USAGE_']:
            Xtrain.append(numpy.array(val,'float32'))
            trainy.append(row['_EMOTIONS_'])
        elif  '_publictest_'    in row['_USAGE_']:
            Xtest.append(numpy.array(val,'float32'))
            testy.append(row['_EMOTIONS_'])
    except:
        print(f"error occured at index :{index} and row:{row}")    

number_features=64
number_labels=7
Batchsize=64
Epochs=200
width,height=48,48

Xtrain = numpy.array(Xtrain,'float32')
trainy=numpy.array(trainy,'float32')
Xtest=numpy.array(Xtest,'float32')
testy=numpy.array(testy,'float32')


trainy=np_utils.to_categorical(trainy,num_classes=number_labels)
testy=np_utils.to_categorical(testy,num_classes=number_labels)

Xtrain -= numpy.mean(Xtrain, axis=0)
Xtrain /= numpy.std(Xtrain,axis=0)

Xtest -= numpy.mean(Xtest, axis=0)
Xtest /= numpy.std(Xtest,axis=0)

Xtrain = Xtrain.reshape(Xtrain.shape[0], 48, 48, 1)
Xtest = Xtest.reshape(Xtest.shape[0], 48, 48, 1)


modelss=Sequential()

modelss.add(Conv2D(64, kernel_size=(3, 3), activation='relu',input_shape=(Xtrain.shape[1:])))
modelss.add(Conv2D(64,kernel_size=(3,3), activation='relu'))

modelss.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
modelss.add(Dropout(0.5))

modelss.add(Conv2D(64, (3,3), activation='relu'))
modelss.add(Conv2D(64,(3,3), activation='relu'))

modelss.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
modelss.add(Dropout(0.5))

modelss.add(Conv2D(128, (3,3), activation='relu'))
modelss.add(Conv2D(128,(3,3), activation='relu'))

modelss.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

modelss.add(Flatten())


modelss.add(Dense(1024, activation='relu'))
modelss.add(Dropout(0.2))
modelss.add(Dense(1024, activation='relu'))
modelss.add(Dropout(0.2))

modelss.add(Dense(number_labels, activation='softmax'))

modelss.compile(loss=categorical_crossentropy,
                optimizer=Adam(),
                metrics=['accuracy'])


modelss.fit(Xtrain, trainy,
            batch_size=Batchsize,
            epochs=Epochs,
            verbose=1,
            validation_data=(Xtest,testy),
            shuffle=True

)

model_json = modelss.to_json()
with open("MODEL.json", "w") as json:
  json.write(model_json)
modelss.save_weights("MODEL.h5")  