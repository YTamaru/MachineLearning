#fine tuning https://spjai.com/keras-fine-tuning/ https://qiita.com/yottyann1221/items/20a9c8a7a02edc7cd3d1
import numpy as np
import tensorflow as tf
import random as rn
import os
from keras import backend as K
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
rn.seed(0)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1) #スレッド数
tf.set_random_seed(0)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
import keras
import numpy as np
import os

input_shape = (224, 224, 3)
batch_size = 128
epochs = 12
num_classes = 2


x = []
y = []
for f in os.listdir("bath"):
    x.append(image.img_to_array(image.load_img("bath/"+f, target_size=input_shape[:2])))    #image_to_array Python Imaging Library to 3d-ndarray
    y.append(0) #load_img("path", target_size=resize) 
for f in os.listdir("kitchen"):
    x.append(image.img_to_array(image.load_img("kitchen/"+f, target_size=input_shape[:2]))) #load_img file img to Python Imaging Library
    y.append(1)

x = np.asarray(x)
x = preprocess_input(x) #image preprocess
y = np.asarray(y)

y = keras.utils.to_categorical(y, num_classes) # y to binary matrix

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state= 3) #separate testdata, traindata

from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16

base_model = VGG16(weights='imagenet', include_top=False) #VGG16 Neural Network Trained Model
x = base_model.output
x = GlobalAveragePooling2D()(x) #input data to easily handling data
x = Dense(1024, activation='relu')(x) # Dense = Fully Connected Neural Network, activation function = the function between nodes
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions) # after fine tuning(retrain weight), remake Model  

for layer in base_model.layers:
    layer.trainable = False # only full connected layers are trained. feedforward networks


model.compile(loss=keras.losses.categorical_crossentropy, # set training process
              optimizer="rmsprop", #loss(gosasaisyouka) = cross entropy, optimizer(saitekika) = rmsprop
              metrics=['accuracy']) #metrics(hyoukakannsu)

history = model.fit(x_train, y_train, #start training x=traindata y=teachingdata
            batch_size=batch_size,
            epochs=epochs, #traing times
            verbose=1,
            validation_data=(x_test, y_test)) #test data

print(model.evaluate(x = x_train, y = y_train, batch_size=batch_size, verbose=1))

model.summary() #max pooling = output is the maximum of the input