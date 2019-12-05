from pelee_net import PeleeNet
from tensorflow.keras.models import load_model

# from keras import backend as K
import cv2, os
import numpy as np
import time

class_names = ['GreenLight', 'RedLight', 'gl', 'rl', 'non']


#define the network
use_stem_block= False
input_shape = (224,224,3)  if use_stem_block else (32,32,3)
model = PeleeNet(input_shape=input_shape, use_stem_block=use_stem_block, n_classes=5)
model = load_model('keras_model_tlseoul.h5')

total_gr =0
total_r = 0
total_grl =0
total_rl = 0
total_none = 0

green_dir = '/data3/nghia_data/dataset/HM320/crop/Green/'
greennames = os.listdir(green_dir)
for name in greennames:
  try:
    image = cv2.imread(green_dir + name)

    image = cv2.resize(image, (32,32))
    np_image_data = np.asarray(image, np.float32)/255
    np_final = np.expand_dims(np_image_data,axis=0)

    predict = model.predict(np_final)

    class_id = np.argmax(predict)
    if class_id==0:
        total_gr+=1
    elif class_id==1:
        total_r+=1
    elif class_id==2:
        total_grl+=1
    elif class_id==3:
        total_rl+=1
    elif class_id==4:
        total_none+=1

    class_name = class_names[class_id]
    
    print(class_name)
  except:
    pass
print('total image', len(greennames))
print('total predicted green: ', total_gr, total_gr/len(greennames)) #green: 938/939; 
print('total predicted red: ', total_r, total_r/len(greennames)) #red: 712/1045; 
print(total_rl, total_grl, total_none)