
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap, SpectralEmbedding, LocallyLinearEmbedding, TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# from cuml.manifold import TSNE as TSNE_CUDA

import pandas as pd
import numpy as np
from tqdm import tqdm


class DimensonalityReduction:
    
    def __init__(self, X, y, random_state=None, scaler=True, test_size=0.5, split_data=True):
        self.X = X
        self.y = y
        
        if split_data is True:    
            self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(
                test_size=test_size,
                random_state=random_state, 
                scaler=scaler            
                )
        else:
            self.X_train = self.X
            self.X_test = self.X
            self.y_train = self.y
            self.y_test = self.y
            
    def split_data(self, test_size=0.5, random_state=None, scaler=True):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, 
            self.y, 
            test_size=test_size, 
            random_state=random_state
        )
        
        if scaler is True:
            self.scaler = StandardScaler()        
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            
        return self.X_train, self.X_test, self.y_train, self.y_test            

    def apply_PCA(self, device='CPU', n_components=2):
        """
        Applies Principal Component Analysis (PCA) to reduce the dimensionality of the training and test data.

        Parameters
        ----------
        - n_components (int): The number of components to keep. Default is 2.

        Returns
        -------
        - X_train_reduced (ndarray): The reduced training data.
        - X_test_reduced (ndarray): The reduced test data.
        """
        
        pca = PCA(n_components=n_components)
        self.X_train_reduced = pca.fit_transform(self.X_train)
        self.X_test_reduced = pca.transform(self.X_test)
        
        return self.X_train_reduced, self.X_test_reduced

    def apply_LDA(self, device='CPU', n_components=2):
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        self.X_train_reduced = lda.fit_transform(self.X_train, self.y_train)
        self.X_test_reduced = lda.transform(self.X_test)
        
        return self.X_train_reduced, self.X_test_reduced

    def apply_KernelPCA(self, device='CPU', kernel='linear', n_components=2):
        kpca = KernelPCA(kernel=kernel, n_components=n_components)
        self.X_train_reduced = kpca.fit_transform(self.X_train)
        self.X_test_reduced = kpca.transform(self.X_test)
        
        return self.X_train_reduced, self.X_test_reduced

    def apply_Isomap(self, device='CPU', n_components=2):
        iso = Isomap(n_components=n_components, n_neighbors=10)
        self.X_train_reduced = iso.fit_transform(self.X_train)
        self.X_test_reduced = iso.transform(self.X_test)
        
        return self.X_train_reduced, self.X_test_reduced
    
    def apply_Localembed(self, device='CPU', n_components=2):
        locle = LocallyLinearEmbedding(n_components=n_components)
        self.X_train_reduced = locle.fit_transform(self.X_train)
        self.X_test_reduced = locle.transform(self.X_test)
        
        return self.X_train_reduced, self.X_test_reduced
    
    def apply_SpectEmbed(self, device='CPU', n_components=2):
        spec = SpectralEmbedding(n_components=n_components)
        self.X_train_reduced = spec.fit_transform(self.X_train)
        self.X_test_reduced = spec.fit_transform(self.X_test)
        
        return self.X_train_reduced, self.X_test_reduced
    
    def apply_Tsne(self, device='CPU', n_components=2):
        """
        Applies t-SNE dimensionality reduction to the training and test data.

        Parameters:
        - device (str): The device to use for computation. Options are 'CPU' and 'CUDA'. Default is 'CPU'.
        - init (str): The initialization method for t-SNE. Default is 'pca'.
        - n_components (int): The number of dimensions in the reduced space. Default is 2.

        Returns:
        - X_train_reduced (array-like): The training data after dimensionality reduction.
        - X_test_reduced (array-like): The test data after dimensionality reduction.
        """

        if device == 'CPU':            
            tsne = TSNE(n_components=n_components, init='pca')
        elif device == 'CUDA':
            tsne = TSNE_CUDA(n_components=n_components, method='fft')
        
        self.X_train_reduced = tsne.fit_transform(self.X_train)
        self.X_test_reduced = tsne.fit_transform(self.X_test)
        
        return self.X_train_reduced, self.X_test_reduced

    def train_test_model(self, model, model_type, X_train, X_test):
        model.fit(X_train, self.y_train)
        y_pred = model.predict(X_test)
        
        return self.model_evaluation(model_type, y_pred)
    
    def model_evaluation(self, model_type, y_pred):
        """Returns accuracy if model is supervised, otherwise returns Rand Index"""
        
        if model_type == 'supervised':
            return accuracy_score(self.y_test, y_pred)
        elif model_type == 'unsupervised':
            return adjusted_rand_score(self.y_test, y_pred)
        
    
    def run_multiple_training(
        self, 
        models, 
        use_reduction=False, 
        reduction_method='PCA', 
        device='CPU', 
        n_components=None, 
        n_runs=1
    ):
        results = []
        
        for name, model in tqdm(models.items(), desc="Overall Progress"):
            evals = []
            
            for _ in tqdm(range(n_runs), desc=f"------{name} : Runs Progress", leave=True):
                if use_reduction:
                    reduction_method_func = getattr(self, f"apply_{reduction_method}")
                    X_train_reduced, X_test_reduced = reduction_method_func(device=device, n_components=n_components) 
                    acc = self.train_test_model(
                        model=model[0], 
                        model_type=model[1],
                        X_train=X_train_reduced,
                        X_test=X_test_reduced
                    )
                else:
                    reduction_method = 'Raw Data'
                    acc = self.train_test_model(
                        model=model[0],
                        model_type=model[1], 
                        X_train=self.X_train, 
                        X_test=self.X_test
                    )
                evals.append(acc)

            average_eval = np.mean(evals)
            std_eval = round(np.std(evals), 4)

            if model[1] == 'supervised':
                results.append(
                    {'model': name, f'{reduction_method} avg acc': average_eval, f'{reduction_method} std acc': std_eval }
                )
            elif model[1] == 'unsupervised':
                results.append(
                    {'model': name, f'{reduction_method} avg rand_score': average_eval, f'{reduction_method} std rand_score': std_eval }
                )
        
        return pd.DataFrame(results)


    
