#Assignment No. 5 - Silhoutte Method
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

df = pd.read_csv("IRIS.csv")

X = df.drop('species', axis = 1)

k_values = range(2, 11)

silhouette_scores = []

for k in k_values:

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    
    silhouette_avg = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)

    print(f"For k = {k}, the average silhouette score is {silhouette_avg:.2f}")

plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.show()


