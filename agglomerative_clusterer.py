import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

data = pd.read_csv("car_sales.csv")

data = data.values[:, [7, 13]]

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')

def show_dendograms():
    plt.figure(figsize=(10, 7))
    plt.title("Dendogram")
    shc.dendrogram(shc.linkage(data, method='ward'))
    plt.show()

def show_plot():
    plt.figure(figsize=(10, 7))
    plt.scatter(data[:,0], data[:, 1], c=cluster.labels_, cmap='rainbow')
    plt.show()

def cluster_data():
    result = cluster.fit_predict(data)
    cluster_1 = []
    cluster_2 = []
    for i in range(len(data)):
        if result[i] == 0:
            cluster_1.append(data[i])
        else:
            cluster_2.append(data[i])
    return cluster_1, cluster_2
