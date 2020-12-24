import pandas as pd
import os
import glob


class PipelineMeta():
    
    
    def __init__(self):
        """
        Initialize class with paths to normal/abnormal .wav files
        """
        self.paths=[
    "data0db/fan/id_00",
    "data0db/fan/id_02",
    "data0db/fan/id_04",
    "data0db/fan/id_06",
    "data6db/fan/id_00",
    "data6db/fan/id_02",
    "data6db/fan/id_04",
    "data6db/fan/id_06",
    "data66db/fan/id_00",
    "data66db/fan/id_02",
    "data66db/fan/id_04",
    "data66db/fan/id_06",
]


    def datasetGenerator(self,
                        targetDir,
                        normalDirName="normal",
                        abnormalDirName="abnormal",
                        ext="wav"):
        """
        For a given path, define path to each .wav, noting information on normal/abnormal
        
        input
        
            targetDir: path to ind. machine
            normalDirName: name for normal .wav files
            abnormalDirName: name for abnormal .wav files
            ext: extension for audio files
            
        output 
        
            normalSet: pandas dataframe contaning file path to all normal .wav audio files of targetDir
            abnormalSet: same as normalSet, but for abnormal .wav
        """
        # 01 normal list generate
        normalFiles = sorted(glob.glob(
            os.path.abspath("{dir}/{normalDirName}/*.{ext}".format(dir=targetDir,
                                                                    normalDirName=normalDirName,
                                                                    ext=ext))))
        normalLabels = False

        # 02 abnormal list generate
        abnormalFiles = sorted(glob.glob(
            os.path.abspath("{dir}/{abnormalDirName}/*.{ext}".format(dir=targetDir,
                                                                    abnormalDirName=abnormalDirName,
                                                                    ext=ext))))
        abnormalLabels = True

        normalSet = pd.DataFrame({"filePath":normalFiles,"label":normalLabels})
        abnormalSet = pd.DataFrame({"filePath":abnormalFiles,"label":abnormalLabels})
        return normalSet, abnormalSet
    
    
    def metaGenerator(self, save=False, metaFileName="Meta.csv"):
        """
        Method to loop through all individual independent machines for all 3 types of added white noise
        
        input
        
            save: Boolean parameter to save output of method (metadata dataframe) as csv
            metaFileName: name for .csv if save=True
            
        output
        
            meta: Dataframe containing filepaths to every .wav in given paths (self.paths from __init__)
        """
        normalSet, abnormalSet = self.datasetGenerator(self.paths[0])
        meta = normalSet.append(abnormalSet) 
        for path in self.paths[1:]:
            normalSet, abnormalSet = self.datasetGenerator(path)
            meta = meta.append(normalSet.append(abnormalSet))
        if save:
            meta.to_csv(metaFileName, index=0)
        return meta
    
        
    def __str__(self):
        return 'class to retrieve filepath and sound type metadata and save as Pandas dataframe'
    
    
if __name__=='__main__':
    meta=PipelineMeta()
    meta.metaGenerator(save=True)