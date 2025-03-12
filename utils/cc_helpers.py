"""
Functions to evaluate the performance metrics for and to plot a channel chart.

@author: Sueda Taner
"""
#%%
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import trustworthiness

#%% Evaluate a channel chart
def evaluate_cc(X: np.array, Y: np.array, metric='TW-CT', plot_eval=False, metric_param=-1):
    """
    Evaluate the quality of a latent space (specifically, the channel chart).
    
    Parameters
    ----------
    X : np.array of size (U, dimension=2 or 3)
        True positions of the UEs in the original space.
    Y : np.array of size (U, dimension=2 or 3)
        Positions of the UEs in the channel chart.
    metric : str, optional
        The evaluation metric. The default is 'TW-CT'. Can be 'KS' or 'RD' too.
    plot_eval : bool, optional
        True to plot TW-CT values. The default is False.
    metric_param : TYPE, optional
        DESCRIPTION. The default is -1.

    Raises
    ------
    Exception
        for undefined evaluation metrics.

    Returns
    -------
        A tuple that is tw, ct or ks, rd
    """
    
    if metric == 'TW-CT':  # trustworthiness - continuity
        tw = trustworthiness(X, Y, n_neighbors = int(0.05 * Y.shape[0]))    
        # Continuity is trustworthiness with swapped inputs
        ct = trustworthiness(Y, X, n_neighbors = int(0.05 * Y.shape[0]))
        return tw, ct
    
    else:
        # Calculate pairwise distances
        d = euclidean_distances(X, X)  # distance between the true positions
        d_tilde = euclidean_distances(Y, Y)  # distance between the points in the latent space
        
        if metric == 'KS-RD':  
            # Kruskal stress
            beta = np.sum(d * d_tilde) / np.linalg.norm(d_tilde, 'fro') ** 2
            ks = np.linalg.norm(d - beta * d_tilde, 'fro') \
                   / np.linalg.norm(d, 'fro')
            
            # Rajski distance
            if metric_param == -1:
                num_bins = 20
            else:
                num_bins = metric_param
            x = d.flatten()
            y = d_tilde.flatten()
            Px, bin_edges_x = np.histogram(x, bins=num_bins)
            Px = Px / len(x)
    
            Py, bin_edges_y = np.histogram(y, bins=num_bins)
            Py = Py / len(y)
    
            Pxy, xedges, yedges = np.histogram2d(x, y, bins=num_bins)
            Pxy = Pxy / len(x)
    
            idcs = np.nonzero(Pxy)
            H = - np.sum(Pxy[idcs] * np.log2(Pxy[idcs]))
            I = np.sum(Pxy[idcs] * np.log2(Pxy[idcs] / (Px[idcs[0]] * Py[idcs[1]])))
    
            assert H != 0
            rd = 1 - I / H
            return ks, rd
        else:
            raise Exception('Undefined evaluation metric')


#%% Plot a channel chart

def plot_chart(X: np.array, colors: np.array, normalize: bool, anchors=None, is_meters=False, lims=None, title=None):
    """
    Plot the channel chart with given colors.

    Parameters
    ----------
    X : np.array
        Positions of the UEs in the channel chart.
    colors : np.array
        RGB colors of the UEs.
    normalize : bool
        Whether the channel chart should be normalized to have zero mean and unit variance. 
    anchors : np.array
        Anchor points. The default is None.
    is_meters : bool, optional
        Whether the x- and y-labels should denote "(m)". The default is True.
    lims : np.array
        x- and y-axis limits. The default is None.
    title : str
        Title of the plot. The default is None.

    """
    fig = plt.figure()
    
    if normalize:
        X = (X - np.mean(X,0,keepdims=True)) / np.std(X,0,keepdims=True)
    
    ax = fig.add_subplot()
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=5)
    ax.set_xlabel('x'), ax.set_ylabel('y')
    if is_meters: ax.set_xlabel('x (m)'), ax.set_ylabel('y (m)')
    if lims is not None:
        ax.set_xlim([lims[0,0], lims[0,1]])
        ax.set_ylim([lims[1,0], lims[1,1]])
        ax.set_xticks([lims[0,0], 0, lims[0,1]]) 
        ax.set_yticks([lims[1,0], 0, lims[1,1]]) 
    ax.axis('equal')
    
    if anchors is not None: ax.scatter(anchors[:,0], anchors[:,1], marker='^')
    
    ax.grid(True)
    if title is not None:
        plt.title(title)
    
    
