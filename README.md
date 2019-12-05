# opencv-Image_classification
Train Image classification on CIFAR10 using tensorflow and keras. Create a inference using opencv. OpenCV to read tensorflow graph.
#train
python pelee_train.py

#convert to tensorflow model
python convert.py

#optimize model for inference
python optimize_inference.py --input tf_model.pb --output opt_model1.pb --input_names input_1_1 --output_names main_output_1/Softmax

#create pbtxt file
python to_prorotxt.py

#predict in opencv
python predict.py
