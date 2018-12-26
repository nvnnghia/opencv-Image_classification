from keras.applications import VGG16
from keras import backend as K
import cv2
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
import numpy as np
import os
from keras.optimizers import RMSprop
import random
from keras.layers import Dense, Dropout, Activation, Conv2D, BatchNormalization, Flatten, Reshape, Permute
from keras.models import Sequential, Model


'''Train network to do image classification, 
    Using Cifar dataset to train and test,
    Cifar dataset have 10 classes,
'''


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#where you store your data
base_dir = 'image/'

#load dataset into memmory
def load_data():
    x_train = []
    y_train = []
    for ind, name in enumerate(class_names):
        name_list = os.listdir(base_dir+name+'/')
        for img_name in name_list:
            image = cv2.imread(base_dir+name+'/'+img_name)
            image = cv2.resize(image, (48,48)) 
            x_train.append(image)
            label = [0]*10
            label[ind]=1
            label = np.asarray(label)
            y_train.append(label)
            
    return x_train, y_train

def get_model():
    model = Sequential()
    #inp = Input(shape=(height, width, 1))                               # input layer 20 x 180
    # [20, 180, 1]
    model.add(Conv2D(32, (5,5), strides=(2, 2), activation='relu', input_shape=(48,48,3)))    # 32 2D convolution layers, kernel 5x5
    # [8, 88, 32]
    #c2 = Conv2D(64, 3, strides=(2, 2), activation='relu')(c1)     # 64 2D convolution layers, kernel 3x3
    model.add(Conv2D(64, (3,3), strides=(2, 2), activation='relu'))
    # [3, 43, 64]
    #c3 = Conv2D(128, 3, strides=(2, 2), activation='relu')(c2)    # 128 2D convolution layers, kernel 3x3
    model.add(Conv2D(128, (3,3), strides=(2, 2), activation='relu'))
    # [1, 21, 128]
    #c4 = Conv2D(256, 3, strides=(2, 2), activation='relu')(c3) 
    model.add(Conv2D(256, (3,3), strides=(2, 2), activation='relu'))

    #f = Flatten()(c4)                           # flatten layer [1, 21, 128] -> [2688]
    #model.add(Flatten())
    #model.add(Reshape((256))
    a,b,c,d = model.output_shape
    a = b*c*d

    model.add(Permute([1, 2, 3]))  # Indicate NHWC data layout
    model.add(Reshape((a,)))
    # [2688]
    #d = Dense(200, activation='relu')(f)        # dense layer
    model.add(Dense(200, activation='relu'))
    # [256]
    #outp = Dense(topclass, activation='softmax')(d)   # dense layer
    model.add(Dense(10, activation='softmax'))
    # [50]

    #model = Model(inputs=inp, outputs=outp)

    #model.compile(optimizer=Adam(lr=1e-3),
    #              loss='categorical_crossentropy',
    #              metrics=['accuracy'])
    #print(model.summary())
    return model


IMAGE_SIZE    = (48, 48)
BATCH_SIZE    = 32  # try reducing batch size or freeze more layers if your GPU runs out of memory
NUM_EPOCHS    = 2
WEIGHTS_FINAL = 'model-vgg16-final_2.h5'

#shuffle the dataset
x_train, y_train = load_data()
c= list(zip(x_train, y_train))
random.shuffle(c)
x_train, y_train = zip(*c)
x_train = np.array(x_train)
y_train = np.array(y_train)
y_train = y_train.astype('float32')
x_train = x_train.astype('float32')
#x_train /= 255

#define the network
#model = VGG16(input_shape=(48, 48, 3), weights=None, include_top=True, classes=10)
model = get_model()

# Compile the model using RMSprop
model.compile(optimizer=RMSprop(lr=0.0001, decay=1e-6), loss='categorical_crossentropy',
                  metrics=['accuracy'])

#summary model information
print(model.summary())

#Train the model
model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHS,
              shuffle=True)

#Save keras model
model.save(WEIGHTS_FINAL)



# convert to tenorflow and save model as .pb file
pred_node_names = [None]
pred = [None]
for i in range(1):
    pred_node_names[i] = "output_node"+str(i)
    pred[i] = tf.identity(model.outputs[i], name=pred_node_names[i])

sess = K.get_session()
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
graph_io.write_graph(constant_graph, ".", "model2.pb", as_text=False)

'''
Class #3 = cat
Class #0 = airplane
Class #6 = frog
Class #2 = bird
Class #7 = horse
Class #1 = automobile
Class #9 = truck
Class #8 = ship
Class #5 = dog
Class #4 = deer
'''
