# Importing all dependencies
# importing the libraries
import os
import numpy as np
# import keras and tensorflow modules
import tensorflow as tf
# Matplotlib to plot test predictions
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Training Class
class faceTrain:
    # initialise faceTrain Class
    def __init__(self):
        self.epochs = 250
        self.state = False

    def test(self):
        if self.state == False:
            print("\n\r Model Not Available")
            return False
        self.load_feature_n_target()
        print("\n\rtesting model...")
        fig = plt.figure(figsize=( 50 , 50 ))

        for i in range( 1 , 6 ):
            sample_image = np.reshape( self.test_feature[i] * 255  , ( 96 , 96 ) ).astype( np.uint8 )
            pred = self.model.predict( self.test_feature[ i : i +1  ] ) * 96
            pred = pred.astype( np.int32 )
            pred = np.reshape( pred[0 , 0 , 0 ] , ( 15 , 2 ) )
            fig.add_subplot( 1 , 10 , i )
            plt.imshow( sample_image.T , cmap='gray' )
            plt.scatter( pred[ : , 0 ] , pred[ : , 1 ] , c='green' )

        print("showing predictions now...")
        plt.show()
        # show predictions on tesing data
        print("testing model... done.")

    # predict locations
    def predict(self,img,model):
        if self.load_model(model):
            img = img / 255
            img = np.reshape( img , ( 96 , 96 ) ).astype(np.uint8)
            predictions = self.model.predict(img) * 96
            predictions = predictions.astype( np.int32 )
            predictions = np.reshape( predictions[ 0 , 0 , 0 ], ( 15 , 2 ) )
            return predictions
        else:
            return False


    # autotrain function
    def auto_train(self):
                # use GPU if available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
        # Restrict TensorFlow to only use the first GPU
            try:
                tf.config.set_visible_devices(gpus[0], 'GPU')
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                print(e)
        self.load_feature_n_target()
        self.train("trained_model")


    # load feature data and target data fron jpeg files
    def load_feature_n_target(self):
        try:
            print("\n\r\n\rloading training and testing data...")

            self.train_feature = np.load("./dataset/x_train.npy")
            self.train_feature = self.train_feature / 255
            print(self.train_feature.shape)
            self.train_target = np.load("./dataset/y_train.npy")
            self.train_target = self.train_target / 96
            self.train_target = np.reshape( self.train_target, ( -1 , 1 , 1 , 30 ))
            print(self.train_target.shape)
            self.test_feature = np.load("./dataset/x_test.npy")
            self.test_feature = self.test_feature / 255
            print(self.test_feature.shape)
            self.test_target = np.load("./dataset/y_test.npy")
            self.test_target = self.test_target / 96
            self.test_target =np.reshape( self.test_target, ( -1 , 1 , 1 , 30 ) )
            print(self.test_target.shape)

            self.image_x = self.train_feature.shape[1]
            self.image_y = self.train_feature.shape[0]
            print("loading training and testing data... done!")
        except:
            print("\n\r Unable to load data \n\r")
            self.state = False
            return False

    # define model ,optimisers , lossfn
    def architecture(self):
        try:
            model_layers = [
            tf.keras.layers.Conv2D( 256 , input_shape=( 96 , 96 , 1 ) , kernel_size=( 3 , 3 ) , strides=2 , activation='relu' ),
            tf.keras.layers.Conv2D( 256 , kernel_size=( 3 , 3 ) , strides=2 , activation='relu' ),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
            tf.keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
            tf.keras.layers.Conv2D( 128 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D( 64 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
            tf.keras.layers.Conv2D( 64 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
            tf.keras.layers.Conv2D( 32 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D( 30 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
            tf.keras.layers.Conv2D( 30 , kernel_size=( 3 , 3 ) , strides=1 , activation='relu' ),
            tf.keras.layers.Conv2D( 30 , kernel_size=( 3 , 3 ) , strides=1 ),
            ]
            self.model = tf.keras.Sequential( model_layers )
        except:
            print("Unable to Initialize model")
            return False
        return True

    # train defined model and fit values
    def train(self, name):
        models = os.listdir("./json_model/")
        if name+".h5" not in models or name+".json" not in models:
            self.state = self.architecture()
            if self.state == False:
                print("Unable to set model architecture! Aborting.")
                return False
            try:
                self.model.compile( loss=tf.keras.losses.mean_squared_error , optimizer=tf.keras.optimizers.Adam( learning_rate=0.0001 ) , metrics=[ 'mse' ] )
            except:
                print("Unable to compile model")
                return False
            print("\n\r---------- training model started ----------")

            try:
                self.model.fit(x=self.train_feature,y=self.train_target,batch_size=50,epochs=self.epochs,validation_data=(self.test_feature,self.test_target))
            except RuntimeError as err:
                print("Error encountered while training")
                print(err)
                return False
            if self.save_model(name) == False:
                print("Unable to save the model to file!!")
            else:
                print("Model saved to disk")

            print("---------- Model training Completed ----------")
        else:
            self.load_model(name)
            return True


    # save defined model
    def save_model(self, name):
        if self.state == False:
            print("Model not available")
            return False
        # save training data
        try:
            json_model = self.model.to_json()
            with open( './json_model/'+name+'.json', 'w' ) as json_file:
                json_file.write(json_model)
            self.model.save_weights('./json_model/'+name+'.h5')
            print(self.model.summary())
            print(f"Saved {name} to drive.")
        except:
            return False
        return True

    # load model from disk
    def load_model(self, name):
        location = ["./json_model/"+name+".json", "./json_model/"+name+".h5"]
        try:
            file = open(location[0])
            self.model = file.read()
            file.close()
            self.model = tf.keras.models.model_from_json(self.model)
            self.model.load_weights(location[1])
            self.state = 1
            print(self.model.summary())
        except:
            print("Model not found. Trying to train model.")
            if self.auto_train():
                print("Model trained and loaded successfully.")
                return True
            else:
                print("Unable to train model!")
                return False

