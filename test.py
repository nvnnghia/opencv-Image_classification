import cv2
import numpy as np

'''
    evaluate image classification using opencv,
    Use opencv to run tensorflow model
'''


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#read tensorflow model
net = cv2.dnn.readNetFromTensorflow('model1.pb', 'model1.pbtxt')

#Read the image to test
image = cv2.imread('image/9_cat.png')

#Define the input
net.setInput(cv2.dnn.blobFromImage(image, size=(48, 48), swapRB=True, crop=False))

#Pass the image to the network and get output
out = net.forward()

#process output
class_id = np.argmax(out)
class_name = class_names[class_id]

#print result

print(class_name)
