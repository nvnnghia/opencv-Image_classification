from pelee_net import PeleeNet
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from PIL import Image
import pickle
import os, cv2
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential([
    sometimes(
        iaa.OneOf([
            iaa.Add((-10, 10), per_channel=0.5),
            iaa.Multiply((0.9, 1.1), per_channel=0.5),
            iaa.ContrastNormalization((0.9, 1.1), per_channel=0.5)
        ])
    ),
    # iaa.Fliplr(0.5),
    iaa.Crop(percent=(0, 0.3)),
    iaa.Flipud(0.5),
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)
],random_order=True)

def myFunc(myimg):
    img = seq.augment_image(myimg)
    return img

def generator(X, y, batch_size, use_augmentation, shuffle, scale):
    if use_augmentation:
        base_gen = keras.preprocessing.image.ImageDataGenerator(
            #vertical_flip=True,
            #width_shift_range=4.0/32.0,
            #height_shift_range=10.0/32.0,
            shear_range=0.2, zoom_range=0.15,
            preprocessing_function=myFunc)
    else:
        base_gen = keras.preprocessing.image.ImageDataGenerator()
    for X_base, y_base in base_gen.flow(X, y, batch_size=batch_size, shuffle=shuffle):
        if scale != 1:
            X_batch = np.zeros((X_base.shape[0], X_base.shape[1]*scale,
                                X_base.shape[2]*scale, X_base.shape[3]), np.float32)
            for i in range(X_base.shape[0]):
                with Image.fromarray(X_base[i].astype(np.uint8)) as img:
                    img = img.resize((X_base.shape[1]*scale, X_base.shape[2]*scale), Image.LANCZOS)
                    X_batch[i] = np.asarray(img, np.float32) / 255.0
        else:
            X_batch = X_base / 255.0
        yield X_batch, y_base

def lr_scheduler(epoch):
    x = 0.4
    if epoch >= 50: x /= 5.0
    if epoch >= 90: x /= 5.0
    if epoch >= 120: x /= 5.0
    return x
def load_data():
    data = []
    labels = []
    # green_dir = '/data3/nghia_data/dataset/HM320/crop/GreenLight/'
    green_dir = '/data3/nghia_data/dataset/TL-Seoul/crop_gt/0/'
    greennames = os.listdir(green_dir)
    for name in greennames:
        image = cv2.imread(green_dir + name)
        image = cv2.resize(image, (64,64))
        data.append(image)
        labels.append([0])

    # red_dir = '/data3/nghia_data/dataset/HM320/crop/RedLight/'
    red_dir = '/data3/nghia_data/dataset/TL-Seoul/crop_gt/1/'
    rednames = os.listdir(red_dir)
    for name in rednames:
        image = cv2.imread(red_dir + name)
        image = cv2.resize(image, (64,64))
        data.append(image)
        labels.append([1])

    grl_dir = '/data3/nghia_data/dataset/TL-Seoul/crop_gt/2/'
    rednames = os.listdir(grl_dir)
    for name in rednames:
        image = cv2.imread(grl_dir + name)
        image = cv2.resize(image, (64,64))
        data.append(image)
        labels.append([2])

    redl_dir = '/data3/nghia_data/dataset/TL-Seoul/crop_gt/3/'
    rednames = os.listdir(redl_dir)
    for name in rednames:
        image = cv2.imread(redl_dir + name)
        image = cv2.resize(image, (64,64))
        data.append(image)
        labels.append([3])

    non_dir = '/data3/nghia_data/dataset/TL-Seoul/non_tl/'
    rednames = os.listdir(non_dir)
    for name in rednames:
        image = cv2.imread(non_dir + name)
        image = cv2.resize(image, (64,64))
        data.append(image)
        labels.append([4])

    X_train, X_test, y_train, y_test = train_test_split( np.asarray(data), np.asarray(labels), test_size=0.1, random_state=42)

    return  X_train, y_train, X_test, y_test


def train(use_augmentation, use_stem_block):
    tf.logging.set_verbosity(tf.logging.FATAL)
    # (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    
    X_train, y_train, X_test, y_test = load_data()
    print(X_train.shape)
    print(X_test.shape)
    # print(type(X_train))
    # print(X_train[0].shape, y_train[0])

    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    # generator
    batch_size = 64
    # scale = 7 if use_stem_block else 1
    scale = 1
    train_gen = generator(X_train, y_train, batch_size=batch_size,
                          use_augmentation=use_augmentation, shuffle=True, scale=scale)
    test_gen = generator(X_test, y_test, batch_size=64,
                         use_augmentation=False, shuffle=False, scale=scale)
    
    # network
    input_shape = (64,64,3) if use_stem_block else (32,32,3)
    model = PeleeNet(input_shape=input_shape, use_stem_block=use_stem_block, n_classes=5)
    model.compile(keras.optimizers.SGD(0.4, 0.9), "categorical_crossentropy", ["acc"])

    scheduler = keras.callbacks.LearningRateScheduler(lr_scheduler)
    hist = keras.callbacks.History()
    print(model.outputs)

    model.fit_generator(train_gen, steps_per_epoch=X_train.shape[0]//batch_size,
                        validation_data=test_gen, validation_steps=X_test.shape[0]//64,
                        callbacks=[scheduler, hist], epochs=150, max_queue_size=1)

    model.save('keras_model_tlseoul64.h5')


if __name__ == "__main__":
    train(True, True) #use_augmentation, use_stem_block

