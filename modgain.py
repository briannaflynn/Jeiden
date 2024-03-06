import jax.numpy as jnp
import jax.ops import index, index_add

def calculate_modularity_gain_vectorized(adj_matrix, communities, node, total_weight):
    num_nodes = adj_matrix.shape[0]
    num_communities = len(jnp.unique(communities))

    # calculate community sums and totals
    community_ids = jnp.arrange(num_communities)
    community_matrix = jnp.zeros((num_nodes, num_communities))
    community_matrix = index_add(community_matrix, index[jnp.arange(num_nodes), communities], 1)
    
    # calculate k_i and identify the community of node i
    k_i = jnp.sum(adj_matrix[node])
    community_of_node = communities[node]

    # get Σ_tot for each community
    community_edges_sum = jnp.zeros(len(jnp.unique(communities)))
    for com in jnp.unique(communities):
        nodes_in_community = jnp.where(communities == com)[0]
        community_edges_sum = community_edges_sum.at[com].set(jnp.sum(adj_matrix[nodes_in_community][:, nodes_in_community]))

    # get the sum of the weights of the edges from node i to nodes in its community k_{i,in}
    nodes_in_i_community = jnp.where(communities == community_of_node)[0]
    k_i_in = jnp.sum(adj_matrix[node, nodes_in_i_community])

    # get Σ_in for node i's community
    sigma_in = community_edges_sum[community_of_node]

    # delta q is the modularity gain for moving node i to a different community
    delta_q = ((sigma_in + 2 * k_i_in) / (2 * total_weight)) - (((community_edges_sum[community_of_node] + k_i) / (2 * total_weight)) ** 2) - \
              ((sigma_in / (2 * total_weight)) - ((community_edges_sum[community_of_node] / (2 * total_weight)) ** 2) - ((k_i / (2 * total_weight)) ** 2))

    return delta_q
