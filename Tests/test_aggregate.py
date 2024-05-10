import jax.numpy as jnp
import numpy as np
import sys
from aggregate import *

def generate_adj_matrix(num_nodes, connectivity='sparse'):
    """
    should probably move this to an __init__ but whatever
    """
    if connectivity == 'dense':
        prob = 0.5  # 50% chance of an edge
    else:
        prob = 0.2  # 20% chance of an edge for sparse

    # generates a symmetric matrix with given probability of edges
    r = np.random.rand(num_nodes, num_nodes)
    r = (r + r.T) / 2  # represents an undirected graph
    adj_matrix = jnp.array(r < prob, dtype=int)

    # zero out the diagonal (no self-loops)
    adj_matrix = adj_matrix.at[jnp.arange(num_nodes), jnp.arange(num_nodes)].set(0)
    return adj_matrix

def test_aggregate_nodes(num_nodes, num_communities, connectivity='sparse'):
    adj_matrix = generate_adj_matrix(num_nodes, connectivity)
    #print(adj_matrix)
    # assign nodes to communities
    communities = jnp.array(np.random.choice(num_communities, num_nodes))
    print('communities\n', communities)
    
    aggregated_adj_matrix, new_communities = aggregate_nodes(adj_matrix, communities)
    print('\n\n')
    print("Adjacency Matrix:\n", adj_matrix)
    print("Communities:", communities)
    print("Aggregated Adjacency Matrix:\n", aggregated_adj_matrix)
    print("New Communities:", new_communities)

print('\n\nTest dense')
test_aggregate_nodes(10, 8, connectivity='dense')

print('\n\nTest sparse')
test_aggregate_nodes(10, 8)
