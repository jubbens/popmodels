from itertools import combinations
import numpy as np
import random


def get_crosses(selected, method='pairs'):
    if method == 'selfing':
        return [(x, x) for x in selected]
    elif method == 'exhaustive':
        return list(combinations(selected, 2))
    elif method == 'pairs':
        random.shuffle(selected)
        p = len(selected) if len(selected) % 2 == 0 else len(selected) - 1
        return [[selected[i], selected[i + 1]] for i in np.arange(0, p, 2)]
    else:
        raise NotImplementedError('unknown crossing strategy: {0}'.format(method))


def phenotypic_selection(pop, n, trait_idx=0, method='pairs', negative=False):
    """Select the n individuals with the highest phenotypic value"""
    pop.sort(key=lambda x: x.phenotypes[trait_idx], reverse=(not negative))
    selected = [p.id for p in pop[:n]]

    crosses = get_crosses(selected, method=method)

    return crosses


def random_selection(pop, n, method='pairs'):
    """Select random individuals"""
    random.shuffle(pop)
    selected = [p.id for p in pop[:n]]

    crosses = get_crosses(selected, method=method)

    return crosses
