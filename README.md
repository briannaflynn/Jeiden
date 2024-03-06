# Jeiden
JAX implentation of the Leiden algorithm for community detection

Main steps of the Leiden algorihtm 

1. Local Moving of Nodes: Nodes are moved to the communities of their neighbors to maximize local modularity.

2. Aggregation of Nodes: Once local moves cannot improve modularity, nodes within the same community are aggregated into a single node, and the first step is repeated

3. Refinement: The communities found in the previous step are refined to distribute nodes more evenly.

For JAX implementation, focusing on vectorization and minimizing explicit loops where possible.

_________________________________________________________________________________________________

## Step 1: Calculating modularity gain from the local moving of nodes to other communities
Goal: Move nodes iteratively towards an optimal community structure

A) Initialize each node to be in its own community.

B) Calculate the modularity gain for moving each node to a different community.

C) Move nodes to optimize modularity, considering the current community assignments.

$\ ΔQ = \left[ \frac{\Sigma_{\text{in}} + 2k_{i,\text{in}}}{2m} - \left( \frac{\Sigma_{\text{tot}} + k_i}{2m} \right)^2 \right] - \left[ \frac{\Sigma_{\text{in}}}{2m} - \left( \frac{\Sigma_{\text{tot}}}{2m} \right)^2 - \left( \frac{k_i}{2m} \right)^2 \right] \$

ΔQ: The change in modularity by moving a node 

Σ in: Sum of the weights of all edges inside the community

_k i, in_: The sum of the weights of the edges from node _i_ to nodes in a candidate community

Σ tot: Total sum of the weights of the edges to nodes in the community

_k i_: Sum of the weights of the edges attacehd to node _i_

_m_: sum of the weights of all edges in the graph
