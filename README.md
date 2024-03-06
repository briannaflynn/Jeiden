# Jeiden
JAX implentation of the Leiden algorithm for community detection

Main steps of the Leiden algorihtm 

1. Local Moving of Nodes: Nodes are moved to the communities of their neighbors to maximize local modularity.

2. Aggregation of Nodes: Once local moves cannot improve modularity, nodes within the same community are aggregated into a single node, and the first step is repeated

3. Refinement: The communities found in the previous step are refined to distribute nodes more evenly.

For JAX implementation, focusing on vectorization and minimizing explicit loops where possible.

