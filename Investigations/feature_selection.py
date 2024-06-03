import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Step 1: Load your radiomics features into a pandas DataFrame
# Replace this with your actual data loading code
# For example, if you have a CSV file containing your features:
# radiomics_data = pd.read_csv('radiomics_features.csv')

# For demonstration purposes, let's create a random dataset
num_samples = 100
num_features = 10
radiomics_data = pd.DataFrame(np.random.rand(num_samples, num_features), columns=[f'Feature_{i}' for i in range(num_features)])

# Step 2: Calculate correlation matrix
correlation_matrix = radiomics_data.corr()

# Step 3: Construct graph
G = nx.from_numpy_matrix(correlation_matrix.values)
G = nx.relabel_nodes(G, dict(zip(range(num_features), correlation_matrix.columns)))

# Step 4: Calculate degree centrality and clustering coefficient for each feature
degree_centrality = nx.degree_centrality(G)
clustering_coefficient = nx.clustering(G)

# Combine degree centrality and clustering coefficient
combined_scores = {feature: degree_centrality[feature] * clustering_coefficient[feature] for feature in G.nodes()}

# Print combined scores for each feature
print("Combined Scores:")
for feature, score in combined_scores.items():
    print(f"{feature}: {score}")

# Plot the graph
pos = nx.spring_layout(G)  # Define node positions using spring layout
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='k', linewidths=0.5, font_size=10)
plt.title('Graph of Radiomics Features')
plt.show()
