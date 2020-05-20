# from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt 
import numpy as np 
import seaborn as sns 
from os import listdir

X = np.load("distance_matrix.npy")

# to show clustering with heatmap -----------------------------------------------------
# sns.clustermap(X, metric="euclidean", standard_scale=1, method="ward", cmap="plasma")
# plt.show()



Z = linkage(X, 'ward')

def leafLabel(idx):
	viblib = listdir("viblib")
	return viblib[idx]

fig = plt.figure(figsize=(25,10))
dn = dendrogram(Z,leaf_label_func=leafLabel)


# link token ids with file names ----



	

# def rename(section):
# 	viblib = listdir("viblib")
# 	for i in range(0,len(dn[section])):
# 		dn[section][i] = viblib[int(dn[section][i])]

# rename('leaves')
# rename('ivl')
plt.savefig("dendorgram_high.png",dpi=300)
print(dn)
plt.show()

