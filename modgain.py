import jax.numpy as jnp
import jax.ops import index, index_add

def calculate_modularity_gain_vectorized(adj_matrix, communities, node, total_weight):
    
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
