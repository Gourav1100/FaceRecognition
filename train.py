# Importing all dependencies
# importing the libraries
import cv2
import pandas as pd
import numpy as np

# import keras and tensorflow modules
from keras.optimizer_v2.adam import Adam
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
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
        self.epochs = 5
        self.load_target_map()
        self.load_feature()
        self.image_x = len(self.feature[0])
        self.image_y = len(self.feature[0][0])

    # load target data and create a map fron csv file
    def load_target_map(self):
        self.labels = list()
        print("loading csv data...")
        self.target = pd.read_csv("./dataset/facial_keypoints.csv")
        self.target_map = {'nose': [],'eyebrow_left': [],'eyebrow_right': [],'eye_left': [],'eye_right': [],'mouth_left': [],'mouth_right': [],'mouth_top': [],'mouth_bottom': []}
        mouth_left = ['mouth_left_corner_x', 'mouth_left_corner_y']
        mouth_top = ['mouth_center_top_lip_x', 'mouth_center_top_lip_y']
        mouth_bottom = ['mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y']
        mouth_right = ['mouth_right_corner_x', 'mouth_right_corner_y']
        eye_left = ['left_eye_center_x', 'left_eye_center_y','left_eye_inner_corner_x', 'left_eye_inner_corner_y', 'left_eye_outer_corner_x', 'left_eye_outer_corner_y']
        eye_right = ['right_eye_center_x', 'right_eye_center_y','right_eye_inner_corner_x', 'right_eye_inner_corner_y', 'right_eye_outer_corner_x', 'right_eye_outer_corner_y']
        eyebrow_left = ['left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y']
        eyebrow_right = ['right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y', 'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y']
        nose = ['nose_tip_x', 'nose_tip_y']
        for i in self.target.keys():
            temp = []
            for j in range(len(self.target[i])):
                temp.append(self.target[i][j])
            self.labels.append(i)
            if i in nose:
                self.target_map['nose'].append(temp)
            elif i in eye_left:
                self.target_map['eye_left'].append(temp)
            elif i in eye_right:
                self.target_map['eye_right'].append(temp)
            elif i in eyebrow_left:
                self.target_map['eyebrow_left'].append(temp)
            elif i in eyebrow_right:
                self.target_map['eyebrow_right'].append(temp)
            elif i in mouth_left:
                self.target_map['mouth_left'].append(temp)
            elif i in mouth_right:
                self.target_map['mouth_right'].append(temp)
            elif i in mouth_top:
                self.target_map['mouth_top'].append(temp)
            elif i in mouth_bottom:
                self.target_map['mouth_bottom'].append(temp)
        print('Loading csv...done')

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

    # load target from target_map
    def load_target(self,id):
        self.target = self.target_map[id]
        self.target = np.array(self.target)
        self.target = self.target.transpose((1,0))

    # split data for training and testing
    def train_test_Split(self):
        self.train_feature,self.test_feature,self.train_target,self.test_target = train_test_split(self.feature, self.target, test_size = 0.1)
        self.train_feature=self.train_feature.reshape(len(self.train_feature),self.image_x,self.image_y,1)
        self.test_feature=self.test_feature.reshape(len(self.test_feature),self.image_x,self.image_y,1)

    # define model ,optimisers , lossfn
    def architecture(self):
        try:
            self.model = True
            self.model = Sequential()
            self.model.add(Conv2D(32,(3,3),input_shape=(self.image_x,self.image_y,1),activation="relu"))
            self.model.add(MaxPooling2D(32,(2,2),padding="same"))
            self.model.add(Conv2D(64,(3,3),activation="relu"))
            self.model.add(MaxPooling2D(64,(2,2),padding="same"))
            self.model.add(Conv2D(64,(3,3),activation="relu"))
            self.model.add(Flatten())
            self.model.add(Dropout(0.3))
            self.model.add(Dense(len(self.target[0]),activation="sigmoid"))
        except:
            print("Unable to Initialize model")
            return False
        return True

    # train defined model and fit values
    def train(self):
        for i in self.target_map.keys():
            print(f"---------- training model for {i} ----------")
            self.load_target(i)
            self.state = self.architecture()
            if self.state == False:
                print("Unable to set model architecture! Aborting.")
                return False
            self.train_test_Split()
            # create augmented images
            aug_data_gen=ImageDataGenerator(width_shift_range=0,height_shift_range=0,zoom_range=0,shear_range=0,rotation_range=0)
            aug_data_gen.fit(self.train_feature)
            self.model.compile(Adam(learning_rate=0.001),loss="categorical_crossentropy",metrics=["accuracy"])
            self.model.fit(aug_data_gen.flow(self.train_feature,self.train_target,batch_size=128),epochs=self.epochs,validation_data=(self.test_feature,self.test_target))
            if self.save_model(i) == False:
                print("Unable to save the model to file!!")
                break
            print(f"---------- {i} Model trained and saved ----------")

    # save defined model
    def save_model(self,name):
        # Lets save training data to json file to reduce runtime at next run
        try:
            json_model = self.model.to_json()
            with open( './json_model/'+name+'.json', 'w' ) as json_file:
                json_file.write(json_model)
            self.model.save_weights('./json_model/'+name+'.h5')
            print(self.model.summary())
            print(f"Saved {name} model to drive")
        except:
            return False
        return True


test = faceTrain()
test.train()
