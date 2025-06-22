import networkx as nx
from torch_geometric.utils.convert import from_networkx


# invoke RNG to generate random int only when a non-empty range is supplied
def conditional_randint(random_state, n_min, n_max):
    if n_min != n_max:
        n = random_state.randint(n_min, n_max + 1)
    else:
        n = n_min
    return n


# invoke RNG to generate random float only when a non-empty range is supplied
def conditional_rand(random_state, n_min, n_max):
    if n_min != n_max:
        n = random_state.uniform(n_min, n_max)
    else:
        n = n_min
    return n


def erdos_renyi_generator(rng, n_min=100, n_max=100, p_min=0.15, p_max=0.15):
    n = conditional_randint(rng, n_min, n_max)
    p = conditional_rand(rng, p_min, p_max)
    G = nx.erdos_renyi_graph(n, p)
    return from_networkx(G)


def barabasi_albert_generator(rng, n_min=100, n_max=100, m_min=4, m_max=4):
    n = conditional_randint(rng, n_min, n_max)
    m = conditional_randint(rng, m_min, m_max)
    G = nx.barabasi_albert_graph(n, m)
    return from_networkx(G)
