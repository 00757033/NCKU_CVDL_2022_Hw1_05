import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator


import tensorflow as tf #tf.keras.datasets.cifar10.load_data()
import random

num_classes = 10
epochs = 30
batch_size = 10

saveDir = os.path.join(os.getcwd(), 'saved_models')
modelName = "keras_VGG19_cifar10_trained_model.h5"
labelMap = { 0 : "airplain", 1 : "automobile", 2 : "bird", 3 : "cat", 4 : "deer", 5 : "dog",6 : "frog", 7 : "horse", 8 : "ship", 9 : "truck"} 

# load cifar10 data
(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()

model = tf.keras.applications.vgg19.VGG19(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(32,32,3),
    pooling=None,
    classes=10,
    classifier_activation='softmax'
)

def load_image():
    # random the value 
    rand = random.randint(1,5000)
    labelY = yTrain[rand][0]
    title = labelMap[labelY]

    plt.imshow(xTrain[rand]) 
    plt.title (title, fontsize=12) 
    plt.axis("off")
    plt.show()

def load_image_ui():

    # random the value 
    rand = random.randint(1,5000)

    labelY = yTrain[rand][0]
    title = labelMap[labelY]

    return xTrain[rand], title
 
def show_training_img():

    # random 9 pic
    array = [ random.randint(1, 5000) for i in range(10)]
    
    for i in range(9):

        title = labelMap[yTrain[array[i]][0]]

        plt.subplot(330 + 1 + i)        
        plt.imshow(xTrain[array[i]])
        plt.title (title, fontsize=10) 
        plt.axis("off")

    plt.show()

def print_model():
    model.summary()

def Show_Data_Augmentation(img):

        testImg = img.copy()
        testImg = cv2.cvtColor(testImg, cv2.COLOR_BGR2RGB)  
        testImg = np.expand_dims(testImg, axis=0)

        #Data Augmentation
        datagen = ImageDataGenerator(   
            rotation_range=30,
            horizontal_flip=True,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.2, 
        )
        datagen.fit(testImg)
        
        augmentationList= []
        
         # three Time
        for i in range(3):  
            for img in datagen.flow(testImg, batch_size=1):
                augmentationList.append(img[0].astype("uint8"))   
                break   

        for i in range(3):
            plt.subplot(131 + i)        
            plt.imshow(augmentationList[i])
        plt.axis("off")
        plt.show()

def test(img):

    # model = tf.keras.models.load_model("saved_models/keras_VGG19_cifar10_trained_model.h5")
    testImg = img.copy()
    testImg = cv2.cvtColor(testImg, cv2.COLOR_BGR2RGB)  
    testImg=cv2.resize(testImg,(32,32),interpolation=cv2.INTER_NEAREST)
    print(testImg.shape)

    model = tf.keras.models.load_model("CIFAR10_model.h5")
    testImg=testImg.astype('float32')/255.0 
       
    testImg = np.expand_dims(testImg, axis=0)
    predictions = model.predict(testImg)

    return predictions.max(),np.argmax(predictions, axis=1)

if __name__ == '__main__':
    print("hw1 5")
    # load_image()
    # show_Cifar10_training_img()
    # print_model()
    # Show_Data_Augmentation()
    test()



