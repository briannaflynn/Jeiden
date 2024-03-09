from jax import jit
import jax.numpy as jnp
# update this later
from modgain import calculate_modularity_gain_vectorized
from aggregate import *

def jeiden_algorithm_step(adj_matrix, communities, total_weight, max_iter, leiden_resolution):
    
    # Base case for recursion could be defined here something like set a `max_iter` value or set based on change in modularity relative to a selected leiden resolution

    # Calculate modularity gain and move nodes
    new_communities, modularity_gain = calculate_modularity_gain_vectorized(adj_matrix, communities, total_weight)

    # Check if the modularity gain is significant compared to leiden_resolution
    if max_iter <= 0 or modularity_gain < leiden_resolution:
        return adj_matrix, communities, current_modularity
    
    # Aggregate nodes based on the new community assignments
    # new_adj_matrix, new_communities = aggregate_nodes(adj_matrix, new_communities)
    
    # Recursively apply the algorithm on the aggregated graph
    # if not base_case:
    #     return jeiden_algorithm_step(new_adj_matrix, new_communities, total_weight, rng_key, max_iter-1)
    # else:
    #     return new_adj_matrix, new_communities

# Initial call to the function
# rng_key = jax.random.PRNGKey(seed_value)
# final_adj_matrix, final_communities = leiden_algorithm_step(initial_adj_matrix, initial_communities, total_weight, rng_key)
