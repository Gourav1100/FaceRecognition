# Importing all dependencies
# importing the libraries
import cv2
import pandas as pd
import numpy as np

# import keras and tensorflow modules
from keras.optimizer_v2.adam import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

# for reading
from skimage.io import imread
from skimage.transform import resize

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score

# Training Class
class faceTrain:
    # initialise faceTrain Class
    def __init__(self):
        self.load_target()
        self.load_feature()
        self.image_x = len(self.feature[0])
        self.image_y = len(self.feature[0][0])
        self.train_test_Split()
        self.state = self.architecture()
        print(self.state)
        if self.state == False:
            print("Aborting")

    # load target data fron csv file
    def load_target(self):
        self.labels = list()
        print("loading csv data...")
        self.target = pd.read_csv("./dataset/facial_keypoints.csv")
        target = list()
        for i in self.target.keys():
            temp = []
            for j in range(len(self.target[i])):
                temp.append(self.target[i][j])
            self.labels.append(i)
            target.append(temp)
        self.target = np.array(target)
        self.target = self.target.transpose((1,0))

    # load feature data fron jpeg files
    def load_feature(self):
        self.feature = list()
        print("loading image data...")
        for img_name in range(len(self.target)):
            image_path = './dataset/jpegs/face_' + str(img_name) + '.jpeg'
            if img_name%1000 == 0:
                print("loading: [ "+str(img_name)+" - "+ str(img_name+1000)+" ]")
            img = imread(image_path)
            img =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = img/255
            # resizing the image to (224,224,3)
            img = resize(img, output_shape=(96,96,1), mode='constant', anti_aliasing=True)
            # converting the type of pixel to float 32
            img = img.astype('float32')
            # appending the image into the list
            self.feature.append(img)
        print("loading...done")
        self.feature = np.array(self.feature)

    # split data for training and testing
    def train_test_Split(self):
        self.train_feature,self.test_feature,self.train_target,self.test_target = train_test_split(self.feature, self.target, test_size = 0.1)
        self.train_feature=self.train_feature.reshape(len(self.train_feature),self.image_x,self.image_y,1)
        self.test_feature=self.test_feature.reshape(len(self.test_feature),self.image_x,self.image_y,1)

    # define model ,optimisers , lossfn
    def architecture(self):
        try:
            self.model = Sequential()
            self.model.add(Conv2D(64,(7,7),input_shape=(self.image_x,self.image_y,1),activation="relu"))
            self.model.add(MaxPooling2D(64,(2,2),padding="same"))
            self.model.add(Conv2D(128,(4,4),activation="relu"))
            self.model.add(MaxPooling2D(64,(2,2),padding="same"))
            self.model.add(Conv2D(128,(4,4),activation="relu"))
            self.model.add(Flatten())
            self.model.add(Dense(30,activation="sigmoid"))
        except:
            print("Unable to Initialize model")
            return False
        return True

    # train defined model and fit values
    def train(self):
        # create augmented images
        aug_data_gen=ImageDataGenerator(width_shift_range=0,height_shift_range=0,zoom_range=0,shear_range=0,rotation_range=0)
        aug_data_gen.fit(self.train_feature)
        if self.state == True:
            self.model.compile(Adam(learning_rate=0.001),loss="categorical_crossentropy",metrics=["accuracy"])
            self.model.fit(aug_data_gen.flow(self.train_feature,self.train_target,batch_size=128),epochs=10)
        else:
            print("Aborting! Model not initialised.")

    # save defined model
    def save_model(self):
        # Lets save training data to json file to reduce runtime at next run
        json_model = self.model.to_json()
        with open( './json_model/trained_model.json', 'w' ) as json_file:
            json_file.write(json_model)
        self.model.save_weights('./json_model/trained_model.h5')
        print(self.model.summary())
        print("Saved model to drive")
        print("saved")

    # test model on testing feature and target
    def test(self):
        print("testing")

test = faceTrain()
test.train()
test.test()
