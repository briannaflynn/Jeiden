import jax.numpy as jnp

def aggregate_nodes(adj_matrix, communities):
  """
    Aggregates nodes within the same community into single nodes to create a condensed graph.
    
    Parameters:
    - adj_matrix (jax.numpy.ndarray): The adjacency matrix of the graph.
    - communities (jax.numpy.ndarray): Array indicating the community of each node.
    
    Returns:
    - jax.numpy.ndarray: The adjacency matrix of the condensed graph, where each node represents a community.
    - jax.numpy.ndarray: The new community assignments for the condensed graph.
    """

  # first get the number of unique communities and new community assignments
  # new community assignments correspond to the index of each unique community in the aggregated graph - index the 'super-nodes'
  unique_communities, inverse_indices = jnp.unique(communities, return_inverse = True)
  num_communities = len(unique_communities)
  new_communities = jnp.arange(num_communities)
  
  # initialize matrix with zeros then fill matrix where rows are nodes and columns are communitiies
  community_matrix = jnp.zeros((len(communities), num_communities)).at[jnp.arange(len(communities)), inverse_indices].set(1)
  
  # aggregate the adjacency matrix by multiplying the transpose of the community matrix to the adjacency matrix and again with the original community matrix
  aggregated_adj_matrix = community_matrix.T @ adj_matrix @ community_matrix

  return aggregated_adj_matrix, new_communities
