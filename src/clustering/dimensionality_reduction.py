from matplotlib.pyplot import figure
import matplotlib.pytplot as plt 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np 


def dimensionality_reduction(X,method='TSNE'):
    scaler = StandardScaler()
    reducer = PCA(random_state=824)
    X_reduced = reducer.fit_transform(scaler.fit_transform(X))
    if method== 'PCA':
        return X_reduced 
    return TSNE(n_components=2, perplexity=30).fit_transform(X_reduced[:,:50])


def visualize_clusters(X_reduced_tsne, y):

    figure(figsize=(6, 5), dpi=200)

    for i in [1,-1]:
        mask, = np.where(y==i)
        plt.scatter(
            X_reduced_tsne[mask,0],
            X_reduced_tsne[mask,1],
            alpha=0.3, 
            s=0.5,
            label= 'supicious' if i == -1 else 'normal'
        )
    leg = plt.legend()
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    plt.show()