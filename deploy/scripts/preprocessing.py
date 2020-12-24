import librosa
import numpy as np


def extractFeatures(file_name):
   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = 2
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

        # Clean-up if needed
        # for var in ['audio','sample_rate','pad_width','file_name']:
        #     del globals()[var]
        # del globals()['var']
        
    except Exception as e:
        print(e,"\n","Error encountered while parsing file: ", file_name)
        return None 
    
    return mfccs