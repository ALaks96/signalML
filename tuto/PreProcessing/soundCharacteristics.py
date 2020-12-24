import struct
import pandas as pd
from metaPipeline import PipelineMeta


class WavFileHelper(PipelineMeta):
    
    
    def __init__(self, metaFileName='Meta.csv'):
        """
        Initialize class, try to load meta.csv containing filepath metadata 
        Upon failure inherit from PipelineMeta() and generate missing metadata
        
        input
        
            metaFileName: Name of saved metadata dataframe
        """
        try:
            self.wavMeta = pd.read_csv(metaFileName)
        except Exception as e:
            print(e, "\n", "meta.csv does not exist, generating it.")
            super().__init__()
            metaObj =  PipelineMeta()
            self.wavMeta = metaObj.metaGenerator(save=True)
    
    
    def FileProperties(self, filename):
        """
        Using struct library extract basic properties of given .wav file. 
        
        input
        
            filename: name of .wav to inspect
            
        output
        
            numChannels: Number of channels used to record audio
            sampleRate: Sample rate of audio
            bitDepth: Bit depth of audio file
        """
        waveFile = open(filename,"rb")
        riff = waveFile.read(12)
        fmt = waveFile.read(36)
        numChannelsString = fmt[10:12]
        numChannels = struct.unpack('<H', numChannelsString)[0]
        sampleRateString = fmt[12:16]
        sampleRate = struct.unpack("<I",sampleRateString)[0]
        bitDepthString = fmt[22:24]
        bitDepth = struct.unpack("<H",bitDepthString)[0]
        return (numChannels, sampleRate, bitDepth)
    
    
    def readFileProperties(self):           
        audioData = []
        for index, row in self.wavMeta.iterrows():
            fileName=row['filePath']
            data = self.FileProperties(fileName)
            audioData.append(data)
        audioDf = pd.DataFrame(audioData, columns=['numChannels','sampleRate','bitDepth'])
        numChannels = audioDf.numChannels.value_counts(normalize=True)
        sampleRate = audioDf.sampleRate.value_counts(normalize=True)
        bitDepth = audioDf.bitDepth.value_counts(normalize=True)
        characteristics = {
            "number of channels":numChannels,
            "sample rate":sampleRate,
            "bit depth":bitDepth
        }
        return characteristics
    
    
    def __str__(self):
        return 'Class to extract .wav properties (sample rate, num channels, bit depth)'
    
    
if __name__=="__main__":
    audioProp = WavFileHelper()
    audioProp.readFileProperties(metaFileName="meta.csv")