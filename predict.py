import cv2, os
import numpy as np

'''
    evaluate image classification using opencv,
    Use opencv to run tensorflow model
'''


class_names = ['GreenLight', 'RedLight','GreenLeft', 'RedLeft', 'None']

#read tensorflow model
net = cv2.dnn.readNetFromTensorflow('opt_model1.pb', 'model1.pbtxt')


image = cv2.imread('data/7.jpg')
image = cv2.resize(image,(32,32))
image = np.asarray(image, np.float32)/255
net.setInput(cv2.dnn.blobFromImage(image, size=(32, 32), swapRB=False, crop=False))
out = net.forward()
class_id = np.argmax(out)
class_name = class_names[class_id]
print(out)

# green_dir = '/data3/nghia_data/dataset/HM320/crop/RedLight/'
# greennames = os.listdir(green_dir)

# total_gr =0
# total_r = 0
# for name in greennames:
#     image = cv2.imread(green_dir + name)
#     # image = cv2.resize(image,(224,224))
#     image = np.asarray(image, np.float32) / 255
#     # cv2.imwrite('b.jpg', image)
#     # image = cv2.imread('b.jpg')

# #Define the input
#     net.setInput(cv2.dnn.blobFromImage(image, size=(224, 224), swapRB=False, crop=False))

#     #Pass the image to the network and get output
#     out = net.forward()

#     #process output
#     class_id = np.argmax(out)
#     if class_id==0:
#         total_gr+=1
#     else:
#         total_r+=1
#     class_name = class_names[class_id]

#     #print result

#     print(class_name)

# print('total image', len(greennames))
# print('total predicted green: ', total_gr, total_gr/len(greennames)) #green: 938/939; 
# print('total predicted red: ', total_r, total_r/len(greennames)) #red: 712/1045; 