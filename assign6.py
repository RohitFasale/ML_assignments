#Assignment No. 6 - Elbow Method
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")

X = df.drop('species', axis = 1)

k_values = range(1, 11)

wcss = []

for k in k_values:

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    
    wcss.append(kmeans.inertia_)
    
    print(f"For k = {k}, the within-cluster sum of squares (WCSS) is {kmeans.inertia_:.2f}")

plt.plot(k_values, wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method for Optimal k')
plt.show()
