from keras.applications import MobileNetV2
from keras import backend as K
import cv2, os
import numpy as np
import time

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

IMAGE_SIZE    = (128, 128)
WEIGHTS_FINAL = 'pretrained.h5'

#define the network
model = MobileNetV2(input_shape=(128, 128, 3), weights=None, include_top=True, classes=10)
model.load_weights(WEIGHTS_FINAL, by_name=True, skip_mismatch=True)

image_dir = 'test_split/airplane/'
image_names = os.listdir(image_dir)
print(image_names[1])
start =time.time()
for name in image_names:

    image =cv2.imread(image_dir+name)
    image = cv2.resize(image, (128,128))
    np_image_data = np.asarray(image)
    np_final = np.expand_dims(np_image_data,axis=0)

    predict = model.predict(np_final)

    class_name = class_names[np.argmax(predict)]
    
    print(class_name)
print(time.time()-start)

