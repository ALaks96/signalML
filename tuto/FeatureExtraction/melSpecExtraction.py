from PreProcessing.metaPipeline import PipelineMeta
import pandas as pd
import numpy as np
import librosa


class MelSpectrogram(PipelineMeta):
    
    
    def __init__(self, metaFileName='Meta.csv'):
        """
        Initialize class, try to load meta.csv containing filepath metadata 
        Upon failure inherit from PipelineMeta() and generate missing metadata
        
        input
        
            metaFileName: Name of saved metadata dataframe
        """
        try:
            self.meta = pd.read_csv(metaFileName)
        except Exception as e:
            print(e, "\n", "meta.csv does not exist, generating it.")
            super().__init__()
            metaObj =  PipelineMeta()
            self.meta = metaObj.metaGenerator(save=True)
                
        
    def dataAugmentation(self, audio):
        """
        Method to augment data by:
            - Adding white noise (at random)
            - Rolling audio
            - Stretching audio
        
        input

            audio: Extracted audio from .wav
            
        output
        
            audioWn: White noise added audio
            audioRoll: Rolled audio
            audioStretch: Stretched audio
        """
        # white noise
        wn = np.random.randn(len(audio))
        audioWn = audio + 0.005*wn
        # shifting
        audioRoll = np.roll(audio, 1600)
        # Stretching
        audioStretch = librosa.effects.time_stretch(audio, rate=1)
        
        return audioWn, audioRoll, audioStretch    

    
    def pad(self, mfccs, padWidth=2):
        """
        Classic padding for DL
        
        input

            mfccs: MelSpectrogram of given audio
            padWidth: Width of padding to be applied
        """
        return np.pad(mfccs, pad_width=((0, 0), (0, padWidth)), mode='constant')
    
    
    def mfccGenerator(self, filePath, augment=True):
        """
        Method to extract audio and compute MelSpectrogram for given .wav. 
    
        input
        
            filePath: Path to .wav
            augment: Boolean parameter, if True apply augment() method.
        
        output
        
            if augment:
                mfccs, mfccsWn, mfccsStretch, mfccsRoll: Padded MelSpectrograms
                for original audio & augmented versions
            else:
                mfccs: Padded MelSpectrogram for original audio
        """
        audio, sampleRate = librosa.load(filePath, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sampleRate, n_mfcc=40)
        if augment:
            Wn, Roll, Stretch = self.dataAugmentation(audio)
            mfccsWn = librosa.feature.mfcc(y=Wn, sr=sampleRate, n_mfcc=40)
            mfccsRoll = librosa.feature.mfcc(y=Roll, sr=sampleRate, n_mfcc=40)
            mfccsStretch = librosa.feature.mfcc(y=Stretch, sr=sampleRate, n_mfcc=40)
            return self.pad(mfccs), self.pad(mfccsWn), self.pad(mfccsStretch), self.pad(mfccsRoll)
        else:
            return self.pad(mfccs)
        
    
    def save(self, dataFrame, fileName="totalAugmentedDf.npy"):
        """
        Method to save adequatly numpy arrays of arrays in pandas Dataframe using 
        numpy binary format. (Pandas built-in .to_csv() will show problems)
        
        input

            dataFrame: Pandas DataFrame to save
            fileName: name to give to numpy binary containing DataFrame
        """
        temp = dataFrame.copy().to_numpy()
        np.save(fileName, dataFrame, allow_pickle=True)


    def getMfccs(self, augment=True, save=True):
        """
        Method to extract audio and compute MelSpectrograms recursively on all filepaths
        contained in metadata. 
        
        input
        
            augment: Boolean parameter to be passed to mfccGenerator method() for augmenting data
            save: Boolean parameter to save resulting pandas DataFrame
            
        output
        
            melSpecDf: Pandas DataFrame containing MelSpectrograms, metadata, booleans to 
                       indicate wether data is for model validation and or if data is 
                       an augmented version or not
        """
        features = []
        for i in range(len(self.meta)):
            valData = False
            if i%1000 == 0:
                print(i)
            filePath = self.meta.loc[i,"filePath"]
            if filePath == "data0db/fan/id_06":
                valData=True
            classLabel = self.meta.loc[i,"label"]
            if augment:
                mfccsWn, mfccsRoll, mfccsStretch, mfccs = self.mfccGenerator(filePath, augment=augment)
                augmented = False
                features.append([mfccs, classLabel, filePath, valData, augmented])
                augmented = True
                features.append([mfccsWn, classLabel, filePath, valData, augmented])
                features.append([mfccsRoll, classLabel, filePath, valData, augmented])
                features.append([mfccsStretch, classLabel, filePath, valData, augmented])
            else:
                mfccs = self.mfccGenerator(filePath, augment=augment)
                features.append([mfccs, classLabel, filePath, valData])
        melSpecDf = pd.DataFrame(features, columns=['feature',
                                                    'classLabel',
                                                    'filePath',
                                                    'valData',
                                                    'augmented'])
        if save:
            self.save(melSpecDf)
            
        return melSpecDf


    def __str__(self):
        return 'Class that will load audio data, augment it if paremeter boolean "augment" is set to True, retrieve mfccs for the audio(s), pad them, label them and add meta information (augmented, validation data) as well as save under npy (numpy binary) if parameter boolean "save" is set to True'


if __name__== "__main__":
    melSpectrogram = MelSpectrogram()
    melSpectrogram.getMfccs(augment=True, save=True)