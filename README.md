# Jeiden
JAX implentation of the Leiden algorithm for community detection

Main steps of the Leiden algorihtm 

1. Local Moving of Nodes: Nodes are moved to the communities of their neighbors to maximize local modularity.

```modgain.py```

3. Aggregation of Nodes: Once local moves cannot improve modularity, nodes within the same community are aggregated into a single node, and the first step is repeated

4. Refinement: The communities found in the previous step are refined to distribute nodes more evenly.

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

## Step 2: Aggregate the nodes within the same community into a single node

The aggregation of nodes in the Leiden algorithm involves combining all nodes within the same community into a single 'super node' in a new, aggregated graph.

This step effectively redues the size of the graph while preserving community structure - operate on the condensed graph in next iterations.

1. Aggregate the adjacency matrix: Sum the weights of edges between communities to create a new adjacency matrix representing the condensed graph.

2. Update community assignments: Create a new community assignment array for the aggregated nodes.

The ```aggregate_nodes``` function determines unique communiti8es, mapping each node first. A community matrix is then constructed where each row corresponds to a node an each column a community. This matrix indicates the membership of nodes to communities. The community matrix and adjacency matrix are multiplied to efffectively sum the weights of the edges between communities - two multiplications allow us to aggregate both the rows and columns. New community assignments simply correspond to the index of each unique community in the aggregated graph.

Let's define a few things - 

Let's consider _A_ as the original adjacency matrix of the graph, where _Aij_ represents the weight fo the edge between the nodes _i_ and _j_. 
_C_ is a binary matrix where each row corresponds to a node on the original adjacency graph, and each column is a community. In this matrix, _Cik_ is 1 if node _i_ is in community _k_, otherwise, 0. _A'_ is the adjacency matrix of the aggregated graph, where _A'kl_ represents the **sum** of the weights of the edges between communities _k_ and _l_.

We can express the transformation of input adjacency matrix _A_ to _A'_ like this:

$\ A' = C^{\top} A C \$

$\ C^{\top} A \$ 
is the multiplication of the transpose of the community assignment matrix $\ C^{\top} \$ by the original adjacency matrix $\ A \$. We take the resulting matrix, and multiply by the community matrix $\ C \$ again. This multiplication sums the weights of the edges between the communities, resulting in the adjacncy matrix for the aggregated graph - where each element is now the total weight of the edges between two communities. 



