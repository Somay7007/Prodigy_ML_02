import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("Mall_Customers.csv")

km = KMeans(n_clusters=5)
predicted = km.fit_predict(data[['Annual Income (k$)', 'Spending Score (1-100)']])
#print(predicted)

data['Cluster'] = predicted
print(data)

plt.figure(figsize=(8, 5))
sns.histplot(data['Annual Income (k$)'], bins=15, kde=True, color='skyblue')
plt.title('Distribution Of Annual Income')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Frequency')
plt.show

plt.figure(figsize=(8, 5))
for cluster in sorted(data['Cluster'].unique()):
    cluster_data = data[data['Cluster'] == cluster].sort_values('Annual Income (k$)')
    plt.plot(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'],
             label=f'Cluster {cluster}', marker='o')
plt.title('Spending Score VS Annual Income by Cluster')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Cluster', y='Annual Income (k$)', data=data, palette='Pastel2')
sns.swarmplot(x='Cluster', y='Annual Income (k$)', data=data, color='black', size=4)
plt.xlabel('Cluster')
plt.ylabel('Annual Income (k$)')
plt.grid(True)
plt.show()

inertia = []
K_range = range(1, 11)

# Scale numerical features
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)

# Optimal number of clusters using Elbow method
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method For Optimal No. Of Clusters')
plt.xlabel('No. of Clusters (k)')
plt.ylabel('Inertia (WCSS)')
plt.xticks(K_range)
plt.grid(True)
plt.show()

optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_features)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=features)
cluster_centers_df['Cluster'] = range(optimal_k)

plt.figure(figsize=(10, 6))
sns.heatmap(cluster_centers_df.set_index('Cluster'), annot=True, cmap='coolwarm')
plt.title('Cluster Centers')
plt.show()

sns.pairplot(data, hue='Cluster', palette='viridis', vars=features)
plt.show()


cluster_summary = data.groupby('Cluster')[features].mean().reset_index()
print(cluster_summary)

# Save the clustered data
data.to_csv('clustered_customers.csv', index=False)