import numpy as np
import pandas as pd
from melSpecExtraction import MelSpectrogram
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class GenerateLabel(MelSpectrogram):
    
    
    def __init__(self, dfFileName='totalAugmentedDf.npy'):
        """
        Initialize class to generate artificial multi class labels 
        Wrapping to total DF integrated
        
        input
        
            dfFileName: Name of binary containing previously generated MelSpectrogram DF
        """
        super().__init__()
        try:
            print('this may take a while... loading entire dataframe')
            features = np.load(dfFileName, allow_pickle=True).tolist()
            self.melSpecDf = pd.DataFrame(features, columns=['feature', 'classLabel', 'filePath', 'valData', 'augmented'])
            self.melSpecDf['featureShape'] = [x.shape for x in self.melSpecDf['feature']] 
            self.melSpecDf = self.melSpecDf[self.melSpecDf['featureShape'] == (40,433)]
            print('done! Moving on')
        except Exception as e:
            print(e, "\n", "MelSpectrograms were not saved, will generate/save/load them")
            self.melSpecDf = self.getMfccs(augment=True, save=True)
        indexAbnormal = self.melSpecDf['filePath'].str.contains('abnormal')
        indexAugmented = self.melSpecDf['augmented'] == 0
        self.abnormalMelSpecDf = self.melSpecDf.loc[(indexAbnormal) & (indexAugmented),]    
        self.normalMelSpecDf = self.melSpecDf.loc[(~indexAbnormal),]
        self.augmentedMelSpecDf = self.melSpecDf.loc[(indexAbnormal) & ~(indexAugmented),]
    
    
    def formatData(self): 
        """
        Method to flatten MelSpectrograms for PCA
        
        output
        
            flattened MelSpectrograms for all abnormal/original .wav files 
        """
        features = self.abnormalMelSpecDf.feature.tolist()
        featuresArray = np.array(features)
        nsamples, nx, ny = featuresArray.shape
        
        return featuresArray.reshape((nsamples,nx*ny))
        

    def elbowMethod(self, features):
        """
        Elbow method visualized to determine optimal nb of clusters
        
        input
        
            features: Flattened MelSpectrograms
            
        output

            graph showing intertia per nb of clusters 
        """
        ks = range(1, 10)
        inertias = []
        for k in ks:
            # Create a KMeans instance with k clusters: model
            model = KMeans(n_clusters=k)
            # Fit model to samples
            model.fit(features.iloc[:,:2])         
            # Append the inertia to the list of inertias
            inertias.append(model.inertia_)
        plt.plot(ks, inertias, '-o', color='black')
        plt.xlabel('number of clusters, k')
        plt.ylabel('inertia')
        plt.xticks(ks)
        plt.show()


    def getPCA(self, nComponents=20):
        """
        Method to apply PCA to flattened MelSpectrograms
        Automatically reshapes each array using formatData method
        Prompts user for nb of components to consider for PCA based
        on var/bias trade off for a given nb of PC
        
        input
        
            nComponents: range of PC to consider for choosing var/bias tradeoff
            
        output
        
            n principal components for MelSpectrograms
        """
        self.featuresArrayReshaped = self.formatData()
        self.X_std = StandardScaler().fit_transform(self.featuresArrayReshaped)
        pca = PCA(n_components=nComponents)
        pca.fit_transform(self.X_std)
        features = range(pca.n_components_)
        plt.bar(features, pca.explained_variance_ratio_, color='black')
        plt.xlabel('PCA features')
        plt.ylabel('variance %')
        plt.xticks(features)
        plt.show()
        try:
            self.nPC = input('From the graph displayed below, how many principal components do you want to keep?')
            assert isinstance(self.nPC, int), 'Input an integer please'
        except Exception as e:
            self.nPC = 2
        self.principalComponents = PCA(n_components=self.nPC).fit_transform(self.X_std)
        self.PCA_components = pd.DataFrame(self.principalComponents)
        plt.scatter(self.PCA_components[0], self.PCA_components[1], alpha=.1, color='black')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.show()


    def clusterViz(self, reducedData, clusterObj):
        """
        Method for cluster visualization in 2d
        
        input 
        
            reducedData: Principal components as array
            clusterObj: Previously initialized cluster object
            
        output
            
            2d swarm plot with cluster borders 
        """
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        h = .02     
        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = reducedData[:, 0].min() - 1, reducedData[:, 0].max() + 1
        y_min, y_max = reducedData[:, 1].min() - 1, reducedData[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Obtain labels for each point in mesh. Use last trained model.
        Z = clusterObj.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation='nearest',
                extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                cmap=plt.cm.Paired,
                aspect='auto', origin='lower')

        plt.plot(reducedData[:, 0], reducedData[:, 1], 'k.', markersize=2)
        # Plot the centroids as a white X
        centroids = clusterObj.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='w', zorder=10)
        plt.title('K-means clustering melSpectrograms (PCA-reduced data)\n'
                'Centroids are marked with white cross')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()
        
    
    def distToCentroid(self, labels, distances):
        """
        Method to compute and plot distance of each point to the centroïd of its cluster
        
        input

            labels: list containing cluster label attached to index
            distances: distance from point to centroïd
            
        output
        
            distribution plot of distances from each point to its cluster's centroïd
        """
        self.clustersPCA = pd.DataFrame([list(i) for i in zip(labels,distances)],columns=['cluster','distance'])
        self.clustersPCA['distanceToCluster'] = self.clustersPCA['distance'].apply(lambda x: min(x))
        self.clustersPCA['distToCluster1'] = self.clustersPCA['distance'].apply(lambda x: x[0])
        self.clustersPCA['distToCluster2'] = self.clustersPCA['distance'].apply(lambda x: x[1])
        self.clustersPCA['distToCluster3'] = self.clustersPCA['distance'].apply(lambda x: x[2])
        self.clustersPCA.cluster.replace({0:1, 1:2, 2:3}, inplace=True)
        sns.displot(data=self.clustersPCA, x='distanceToCluster', hue='cluster', kde=True)
        plt.show()


    def viz3d(self):
        """
        Method to visualize a 3D swarm plot of first 3 PCs, colored by cluster
        
        output
        
            3D swarm plot
        """
        pca = PCA(n_components=3)
        try:
            components = pca.fit_transform(self.X_std)
        except Exception as e:
            print(e)
            self.getPCA()
        kmeans = KMeans(init='k-means++', n_clusters=self.nClt)
        kmeans.fit(components)
        kmeans.labels_
        total_var = pca.explained_variance_ratio_.sum() * 100
        fig = px.scatter_3d(
            components, x=0, y=1, z=2, color=kmeans.labels_,
            title=f'Total Explained Variance: {total_var:.2f}%',
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )
        fig.show()

        
    def cluster(self, checkNbClusters=True, visualize=True, checkDist=True, d3=False):
        """
        Method to apply step by step Dimensionality reduction and clustering on first k chosen PCs
        Calls public methods to help drive PCA & Clustering
        
        input

            checkNbClusters: Boolean parameter, if True will call elbowMethod visualization method 
            visualize: Boolean parameter, if True will call clusterViz visualization method
            checkDist: Boolean parameter, if True will call distToCentroid visualization method
            d3: Boolean parameter, if True will call viz3d visualization method
        
        output
        
            multi class labels for abnormal non augmented .wav files
        """
        self.getPCA(20)
        if checkNbClusters:
            self.elbowMethod(features=self.PCA_components)
            try:
                self.nClt = input('From the graph displayed below, how many clusters do you want?')
                assert isinstance(self.nClt, int), 'Input an integer please'
            except Exception as e:
                self.nClt = 3       
        kmeans = KMeans(init='k-means++', n_clusters=self.nClt)
        kmeans.fit(self.principalComponents)
        if visualize and self.nPC==2:
            self.clusterViz(self.principalComponents, kmeans)
        if checkDist:
            distances = kmeans.fit_transform(self.principalComponents)
            self.distToCentroid(kmeans.labels_, distances)
        if d3:
            self.viz3d()
        self.clustersPCA.reset_index(inplace=True, drop=True)
        self.abnormalMelSpecDf.reset_index(inplace=True, drop=True)
        try:
            self.abnormalMelSpecDf['cluster'] = self.clustersPCA['cluster']
        except Exception as e:
            print(e, 'need to generate temp dataframe with clusters')
            distances = kmeans.fit_transform(self.principalComponents)
            self.distToCentroid(kmeans.labels_, distances)
            self.abnormalMelSpecDf['cluster'] = self.clustersPCA['cluster']
            
    
    def getTotalDf(self, save=True):
        """
        Method to obtain final dataframe with adequate multi class labels
        
        input
        
            save: Boolean parameter, if True will save final DF as npy binary
            
        output
        
            multiclass labeled MelSpectrogram DF 
        """
        self.cluster()
        self.augmentedMelSpecDf.reset_index(inplace=True, drop=True)
        self.abnormalMelSpecDf.reset_index(inplace=True, drop=True)
        self.augmentedMelSpecDf = pd.merge(self.augmentedMelSpecDf,self.abnormalMelSpecDf[['filePath','cluster']], on='filePath', how='left')
        self.normalMelSpecDf['cluster'] = 0
        self.normalMelSpecDf.reset_index(inplace=True, drop=True)
        totalDf = self.normalMelSpecDf.append(self.abnormalMelSpecDf.append(self.augmentedMelSpecDf))
        if save:
            self.save(totalDf, fileName="totalAugmentedMultiClassDf.npy")
        return totalDf
    
    
    def __str__(self):
        return 'Class to apply dimensionality reduction and clustering to generate multiclass label for sound dataset'
     
    
if __name__=="__main__":
    labelGen = GenerateLabel()
    totalDf = labelGen.getTotalDf(save=True)