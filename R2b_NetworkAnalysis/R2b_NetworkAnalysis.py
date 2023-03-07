
import numpy as np
import pandas as pd


folder = 'PAPER2\\R2b_NetworkAnalysis\\'

######### Network analysis

from tvb.datatypes import connectivity
import networkx as nx

## Working over the averaged matrices
# Load structures
conn = connectivity.Connectivity.from_file("E://LCCN_Local/PycharmProjects/CTB_data3/subjAVG_AAL2pTh_pass.zip")

matrix = conn.weights
regionLabels = conn.region_labels


# Convert matrices to adj matrices
net = nx.convert_matrix.from_numpy_array(np.asarray(matrix))
    # This generates an undirected graph (Graph). Not a directed graph (DiGraph).


# label mapping
mapping = {i: roi for i, roi in enumerate(regionLabels)}
net = nx.relabel_nodes(net, mapping)

### NETWORK METRICS  # Compute metrics of interest for all nodes: append to dataframe

## Centrality
# 1. Degree normalized
degree = pd.DataFrame.from_dict(nx.degree_centrality(net), orient="index", columns=["degree"])


# 2. Node strength normalized
# node_strength_norm = pd.DataFrame.from_dict({node: val/matrix_single_nodes.sum(axis=1).max()
#                                         for (node, val) in net.degree(weight="weight")},
#                                        orient="index", columns=["node_strength_norm"])

# 2b. Node strength
node_strength = pd.DataFrame.from_dict({node: round(val, 4)
                                        for (node, val) in net.degree(weight="weight")},
                                       orient="index", columns=["node_strength"])

# Specific connectivity Pre-ACC
# matrix_single_nodes[regionLabels.index("Precuneus_L"):regionLabels.index("Precuneus_R")+1, regionLabels.index("Cingulate_Ant_L"):regionLabels.index("Cingulate_Ant_R")+1]
# sum(sum(matrix_single_nodes[regionLabels.index("Precuneus_L"):regionLabels.index("Precuneus_R")+1, regionLabels.index("Cingulate_Ant_L"):regionLabels.index("Cingulate_Ant_R")+1]))

# 3. Closeness
closeness = pd.DataFrame.from_dict(nx.closeness_centrality(net), orient="index", columns=["closeness"])

# 4. Betweeness
betweeness = pd.DataFrame.from_dict(nx.betweenness_centrality(net), orient="index", columns=["betweeness"])


## Global Integration
# 5. Path length
path_length = pd.DataFrame.from_dict({source: np.average(list(paths.values()))
                                      for source, paths in nx.shortest_path_length(net)},
                                     orient="index", columns=["path_length"])
np.std(path_length.values)
nx.average_shortest_path_length(net)

# Specific path length ACC-Pre
nx.shortest_path_length(net, source="Precuneus_L", target="Cingulate_Ant_L")
nx.shortest_path_length(net, source="Precuneus_R", target="Cingulate_Ant_L")
nx.shortest_path_length(net, source="Precuneus_L", target="Cingulate_Ant_R")
nx.shortest_path_length(net, source="Precuneus_R", target="Cingulate_Ant_R")


## Local Segregation
# 6. Clustering coefficient
clustering = pd.DataFrame.from_dict(nx.clustering(net), orient="index", columns=["clustering"])
np.std(clustering.values)
nx.average_clustering(net)

# 7. Modularity (Newman approach)
comms = nx.community.greedy_modularity_communities(net)
nx.community.modularity(net, comms)


#### Gatering Results
network_analysis = pd.concat([degree, node_strength, betweeness, path_length], axis=1).reindex(degree.index)

network_analysis["node_strength"] = network_analysis.node_strength/np.max(network_analysis.node_strength)

network_analysis.to_csv(folder + "NetworkAnalysis.csv")
