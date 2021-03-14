import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams

# Function to get distance with Haversine formula 
def haversine(lat1, lon1, lat2, lon2): 
      
    # distance between latitudes 
    # and longitudes 
    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0
  
    # convert to radians 
    lat1 = (lat1) * math.pi / 180.0
    lat2 = (lat2) * math.pi / 180.0
  
    # apply formula
    a = (pow(math.sin(dLat / 2), 2) + 
         pow(math.sin(dLon / 2), 2) * 
             math.cos(lat1) * math.cos(lat2)); 
    rad = 6371 # Earth radius to get kilometers
    c = 2 * math.asin(math.sqrt(a))
    
    d = rad * c # distance in Kilometers
    
    return d

    

# Function to plot MAE results
def plot_metrics (metrics, models, title):

    length = len(metrics)
    x_labels = models

    # Set plot parameters
    rcParams['figure.figsize'] = 8, 4
    fig, ax = plt.subplots()
    width = 0.2 # width of bar
    x = np.arange(length)

    rects1 = ax.bar(x, metrics[:,0], width, color='#000080', label='MAE imputados')
    rects2 = ax.bar(x + width, metrics[:,1], width, color='#0F52BA', label='MAE no imputados')


    ax.set_ylabel('Minutes')
    ax.set_ylim(0,75)
    ax.set_xticks(x + width + width/2)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Models')
    ax.set_title(title)
    ax.legend()
    plt.grid(True, 'major', 'y', ls='--', lw=.9, c='k', alpha=.3)
    fig.tight_layout()
    plt.show()
    

# Function to plot feature importance
def plot_feature_importance(fit_model, feat_names):
    """
    Plot relative importance of a feature subset given a fitted model.
    """

    # Seteamos el tamaño de nuestro plot
    rcParams['figure.figsize'] = 10, 5
    # Guardamos las columnas de nuestro conjunto de entrenamiento
    features = feat_names
    # Obtenemos la importancia de nuestros atributos desde el modelo entrenado
    importances = fit_model.feature_importances_
    # Ordenamos de mayor a menor la importancia de nuestro atributos
    indices = np.argsort(importances)

    # Graficamos
    plt.title('Feature Importance')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('porcentaje de importancia relativa')
    plt.show()