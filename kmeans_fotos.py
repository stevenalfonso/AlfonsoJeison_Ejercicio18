import glob
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster

files = glob.glob('imagenes/*.png')
data = []
for filename in files:
    im = np.float_(plt.imread(filename).flatten())
    data.append(im)

n_clusters = np.arange(1,20,1)
inertia = []
for i in n_clusters:
    k_means = sklearn.cluster.KMeans(n_clusters = i)
    k_means.fit(data)
    inertia.append(k_means.inertia_)

plt.plot(n_clusters, inertia)
plt.xlabel('n_clusters')
plt.ylabel('inertia')
plt.savefig('inercia.png')
plt.show()

best_n_cluster = 5
k_means = sklearn.cluster.KMeans(n_clusters = best_n_cluster)
k_means.fit(data)
cluster = k_means.predict(data)

for i in range(best_n_cluster):
    centers = k_means.cluster_centers_[i]
    norma = np.zeros(87)
    for j in range(87):
        norma[j] = np.int_(np.linalg.norm(data - centers))
