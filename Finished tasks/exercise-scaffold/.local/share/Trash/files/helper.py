import numpy as np
import pandas as pd
import gif
import matplotlib.pyplot as plt

# @gif.frame
def plot_weights(mlp,epochnum=0):
    n_hidden = 50
    weights = {}
    for i in range(0,5,2):
        weights[i] = mlp.get_weights()[i]
    df = pd.DataFrame(columns= ['Layer 1','Layer 2','Layer 3','y'])
    df['Layer 1'] = np.array(list(weights[0].flatten())*n_hidden).reshape(n_hidden,n_hidden).T.reshape(n_hidden**2,)
    df['Layer 2'] = weights[2].flatten()
    df['Layer 3'] = list(weights[4].flatten())*n_hidden
    df.y = 'Weights'
    with plt.xkcd(scale=0.3):
        fig = plt.figure(figsize=(10,6))
        plt.rcParams.update({'font.size': 16})
        numweights = df[(df['Layer 1'].abs() > 0.1) & (df['Layer 2'].abs() > 0.1)].shape[0]
        pd.plotting.parallel_coordinates(df[(df['Layer 1'].abs() > 0.1) & (df['Layer 2'].abs() > 0.1)], "y",
                                         color=["#1C758A"],
                                         cols = ['Layer 1','Layer 2','Layer 3'],
                                         alpha=0.8,lw=3 ) 
        
        plt.title(f'{numweights} non-zero weights after {epochnum} epochs ')