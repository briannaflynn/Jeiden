import jax.numpy as jnp
import jax.ops import index, index_add

def calculate_modularity_gain_vectorized(adj_matrix, communities, node, total_weight):
    """
    Calculates the modularity gain for each node for moving to each possible community, 
    vectorized for efficiency with JAX. This function avoids explicit loops and conditionals 
    to be compatible with JAX's JIT compilation for high-performance computations.

    Parameters:
    - adj_matrix (jax.numpy.ndarray): The adjacency matrix of the graph, where `adj_matrix[i, j]`
      represents the weight of the edge between nodes `i` and `j`. It should be a square matrix.
    - communities (jax.numpy.ndarray): A 1D array where each element represents the community 
      assignment of the node at the corresponding index.
    - total_weight (float): The sum of all the weights in the adjacency matrix. This is used to
      normalize the modularity gain calculations.

    Returns:
    - jax.numpy.ndarray: A 2D array where element (i, j) represents the modularity gain for node `i`
      if it were to move to community `j`. The diagonal, representing no change in community, 
      is set to negative infinity to exclude it from consideration as a valid move.

    Notes:
    - This function assumes that the graph is undirected and the adjacency matrix is symmetric.
    - The number of unique communities is determined from the `communities` array, and it's
      assumed that communities are contiguous integers starting from 0.
    - It's important to precalculate `total_weight` as it remains constant and its calculation
      can be expensive for large graphs.
    - The returned matrix's diagonal elements are set to -inf to indicate that staying in the 
      current community is not considered a valid move for modularity gain.
    """
    
    # get number of nodes and communities, then community sums and totals
    num_nodes = adj_matrix.shape[0]
    num_communities = len(jnp.unique(communities))
    community_ids = jnp.arrange(num_communities)
    community_matrix = jnp.zeros((num_nodes, num_communities))
    community_matrix = index_add(community_matrix, index[jnp.arange(num_nodes), communities], 1)
    
    # calculate k_i for all nodes in adjacency matrix and get the sum of the weights of the edges from node i to nodes in its community k_{i,in}
    k_i = jnp.sum(adj_matrix, axis=1)
    k_i_in_matrix = jnp.dot(adj_matrix, community_matrix)

    # delta q matrix is the modularity gain for moving node i to a different community
    delta_q_matrix = (sigma_tot + 2 * k_i_in_matrix) / (2 * total_weight) - \
                     ((sigma_tot + k_i[:, None]) / (2 * total_weight)) ** 2 - \
                     (sigma_tot / (2 * total_weight) - (sigma_tot / (2 * total_weight)) ** 2 - (k_i[:, None] / (2 * total_weight)) ** 2)

    # fix delta_q_matrix so as to exclude the current community of the node to ensure no gain is calculated for staying in the same community
    current_community_mask = community_matrix.astype(bool)
    delta_q_matrix = jnp.where(current_community_mask, -jnp.inf, delta_q_matrix)

    return delta_q_matrix
