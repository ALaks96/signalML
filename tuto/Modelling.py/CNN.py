from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint 
import tensorflow as tf
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
import pandas as pd
import numpy as np
from datetime import datetime 

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
from FeatureExctraction.labelGeneration import GenerateLabel
    
    
class DCNN(GenerateLabel):
    
    
    def __init__(self, fileName='totalAugmentedMultiClassDf.npy'):
        """
        Initialize DCNN class to fit model on previously generated multi class MelSpectrogram DF
        If file is save as npy binary will load, otherwise will generate
        """
        try:
            print('Trying to load total df containing multiclass labels and augmented melspectrograms')
            features = np.load('totalAugmentedMultiClassDf.npy', allow_pickle=True).tolist()
            # Convert into a Panda dataframe 
            self.totalDf = pd.DataFrame(features, columns=['feature', 'classLabel', 'filePath', 'valData', 'augmented', 'featureShape', 'cluster'])
            print('Finished feature extraction from ', len(self.totalDf), ' files') 
        except Exception as e:
            print(e, "Augmented multi labeled melspectrograms weren't generated, will create them and save as npy binary")
            super().__init__()
            labelGen = GenerateLabel()
            self.totalDf = labelGen.getTotalDf(save=True)
        indexVal = self.totalDf['valData'] == 0
        self.featuresDf = self.totalDf.copy().loc[indexVal,]
            
    
    def formatData(self):
        """
        Method to reshape exogenous variables & encode multi class labels
        
        output

            Train/test splitted data 
            multi label encoded endogenous variable
        """
        # Convert features and corresponding classification labels into numpy arrays
        X = np.array(self.featuresDf.feature.tolist())
        y = np.array(self.featuresDf.cluster.tolist())
        # Encode the classification labels
        le = LabelEncoder()
        self.yy = to_categorical(le.fit_transform(y)) 
        # split the dataset 
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(X, self.yy, test_size=0.08, random_state = 42)
        
    
    def architecture(self, numRows=40, numColumns=433, numChannels=1, filterSize=2, lr=0.01):
        """
        Method defining & compiling CNN architecture
        
        input

            numRows: Number of rows for input array of arrays (MelSpectrogram)
            numColumns: Number of columns for input array of arrays (MelSpectrogram)
            numChannels: Number of channels used to record .wav
            filterSize: Size of filter applied on neural layer
            lr: Learning rate for adam optimizer
            
        output
            
            Compiled sequential keras model
        """
        self.formatData()
        self.xTrain = self.xTrain.reshape(self.xTrain.shape[0], numRows, numColumns, numChannels)
        self.xTest = self.xTest.reshape(self.xTest.shape[0], numRows, numColumns, numChannels)
        num_labels = self.yy.shape[1]
        # Construct model 
        model = Sequential()
        model.add(Conv2D(filters=16,
                         kernel_size=2,
                         input_shape=(numRows, numColumns, numChannels),
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.2))

        model.add(Conv2D(filters=32,
                        kernel_size=2, 
                        activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.2))

        model.add(Conv2D(filters=64, 
                         kernel_size=2, 
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.2))

        model.add(GlobalAveragePooling2D())
        model.add(Dense(num_labels,
                        activation='softmax'))
        
        optimizer = tf.keras.optimizers.Adam(0.001)
        optimizer.learning_rate.assign(lr)
        # Compile the model
        model.compile(loss='categorical_crossentropy',
                      metrics=['accuracy'],
                      optimizer='adam')
        # Display model architecture summary 
        model.summary()
        # Calculate pre-training accuracy 
        score = model.evaluate(self.xTest, self.yTest, verbose=1)
        accuracy = 100*score[1]
        print("Pre-training accuracy: %.4f%%" % accuracy) 
        
        return model
    
    
    def fit(self, numEpochs=150, numBatchSize=256):
        """
        Method to fit compiled model to train/test data
        
        input
        
            numEpochs: Number of epochs for training
            numBatchSize: Batch size for neural layer 
            
        output
        
            Trained multiclass sequential DCNN, saved to hdf5 @saved_models/weights.best.basic_cnn.hdf5
        """
        model = self.architecture()
        checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.basic_cnn.hdf5', 
                                       verbose=1, 
                                       save_best_only=True)
        start = datetime.now()
        model.fit(self.xTrain,
                  self.yTrain, 
                  batch_size=numBatchSize, 
                  epochs=numEpochs, 
                  validation_data=(self.xTest, self.yTest), 
                  callbacks=[checkpointer],
                  verbose=1)
        duration = datetime.now() - start
        print("Training completed in time: ", duration)
        # Evaluating the model on the training and testing set
        score = model.evaluate(self.xTrain,
                               self.yTrain,
                               verbose=0)
        print("Training Accuracy: ", score[1])
        score = model.evaluate(self.xTest,
                               self.yTest,
                               verbose=0)
        print("Testing Accuracy: ", score[1])
        
        
        def __str__(self):
            return 'Class to train DCNN on MelSpectrograms, augmented, with mutli label'
        
        
if __name__ == "__main__":
    model = DCNN()
    model.fit()