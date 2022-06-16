from sbwrapper import Trial
from breedstrats import random_selection, phenotypic_selection
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
from matplotlib import pyplot as plt
from joblib import dump
from csv import writer
import os
from shutil import rmtree


def generate_random_population(output_dir, do_splitting=True, do_selection=False):
    num_snps = 24000
    num_base = 100
    heritability = [0.5]
    n_qtl = 10
    num_timesteps = 10

    subpopulations = {}
    subpopulation_nonce = 0
    valid_subpops = []

    global_generation_count = 0
    generation_nonce = 0
    generations_graph = nx.DiGraph()
    population_graph = nx.DiGraph()

    experiment = Trial()
    experiment.generate_random_founders(num_snps, num_base, ploidy=2)
    experiment.define_traits(h2=heritability, nqtl=n_qtl)

    experiment.make_founder_generation()
    founders = experiment.get_generation(0)

    # Add founder generation to the graph
    generations_graph.add_node(generation_nonce, ids=[ind.id for ind in founders])

    # Add founders to the population graph
    for ind in founders:
        population_graph.add_node(ind.id, individual=ind)

    subpopulations[subpopulation_nonce] = [generation_nonce]
    valid_subpops.append(subpopulation_nonce)

    for i in range(num_timesteps):
        # Main tick loop
        print('Tick {0}'.format(i))
        new_subpops = []

        # Make the latest generation for all the subpopulations
        for j in valid_subpops:
            print('Doing subpopulation {0}'.format(j))
            youngest_subpop_uid = subpopulations[j][-1]
            youngest_subpop_ids = generations_graph.nodes[youngest_subpop_uid]['ids']
            youngest_subpop = [population_graph.nodes[iid]['individual'] for iid in youngest_subpop_ids]

            if do_selection:
                crosses = phenotypic_selection(youngest_subpop, int(len(youngest_subpop) / 2), method='pairs')
                experiment.make_crosses(crosses, num_children=np.random.randint(2, 9, len(crosses)))
            else:
                crosses = random_selection(youngest_subpop, len(youngest_subpop), method='pairs')
                experiment.make_crosses(crosses, num_children=np.random.randint(1, 5, len(crosses)))

            global_generation_count += 1
            generation_nonce += 1
            current_gen = experiment.get_generation(global_generation_count)

            # Add individuals to population graph
            for ind in current_gen:
                population_graph.add_node(ind.id, individual=ind)
                population_graph.add_edge(ind.dam, ind.id)
                population_graph.add_edge(ind.sire, ind.id)

            # Add subpopulations
            cond = (np.random.rand() > 0.3) if do_splitting else True

            if cond:
                # All individuals are staying in this subpop
                subpopulations[j].append(generation_nonce)
                generations_graph.add_node(generation_nonce, ids=[ind.id for ind in current_gen])
                generations_graph.add_edge(youngest_subpop_uid, generation_nonce)
            else:
                halfers = int(len(current_gen) / 2)
                # Those staying make a new generation in this subpop
                staying = current_gen[:halfers]
                subpopulations[j].append(generation_nonce)
                generations_graph.add_node(generation_nonce, ids=[ind.id for ind in staying])

                generations_graph.add_edge(youngest_subpop_uid, generation_nonce)

                # Those going create a new generation in a new subpop
                subpopulation_nonce += 1
                new_subpops.append(subpopulation_nonce)
                print('Adding subpopulation {0}'.format(subpopulation_nonce))

                generation_nonce += 1
                subpopulations[subpopulation_nonce] = [generation_nonce]

                going = current_gen[halfers:]
                generations_graph.add_node(generation_nonce, ids=[ind.id for ind in going])
                generations_graph.add_edge(youngest_subpop_uid, generation_nonce)

        valid_subpops.extend(new_subpops)

    # Network visualization
    plt.figure()
    nx.draw(generations_graph, graphviz_layout(generations_graph, prog='dot'), with_labels=True)
    plt.title('Generation Tree')
    plt.savefig(os.path.join(output_dir, 'tree.pdf'))

    # Network visualization
    # plt.figure()
    # nx.draw(population_graph, graphviz_layout(population_graph, prog='dot'), with_labels=True)
    # # nx.draw(population_graph, pos=nx.spring_layout(population_graph))
    # plt.title('Family Tree')
    # plt.show(block=False)

    # Make a distance map of individuals
    print('Calculating distances')

    latest_generations_idx = [subpopulations[j][-1] for j in valid_subpops]

    all_inds = []
    subpop_inds = []

    for i in latest_generations_idx:
        all_inds.extend(generations_graph.nodes[i]['ids'])
        subpop_inds.append(generations_graph.nodes[i]['ids'])

    graph_ud = population_graph.to_undirected()

    num_inds = len(all_inds)
    dm_ind = np.full((num_inds, num_inds), np.nan)

    for i in range(len(all_inds)):
        for j in range(len(all_inds)):
            ind1 = all_inds[i]
            ind2 = all_inds[j]

            if nx.has_path(graph_ud, source=ind1, target=ind2):
                sp = nx.shortest_path_length(graph_ud, source=ind1, target=ind2)
                dm_ind[i, j] = sp
                dm_ind[j, i] = sp

    plt.figure()
    plt.imshow(dm_ind)
    plt.title('Path Length Between Individuals')
    plt.savefig(os.path.join(output_dir, 'dm.pdf'))

    # Write subpopulation idxs
    writer(open(os.path.join(output_dir, 'subpops.csv'), 'w+', newline='')).writerows(subpop_inds)

    # Write parents
    parents = []

    for node_number in list(population_graph.nodes):
        cur = population_graph.nodes[node_number]['individual']
        parents.append([cur.id, cur.dam, cur.sire])

    writer(open(os.path.join(output_dir, 'parents.csv'), 'w+', newline='')).writerows(parents)

    # Write genotypes
    output = [[iid, population_graph.nodes[iid]['individual'].genotype] for iid in all_inds]
    dump(output, os.path.join(output_dir, 'simpop_random.bin'))

    # Write distance matrix
    output_ped = [all_inds, dm_ind]
    dump(output_ped, os.path.join(output_dir, 'simpop_random_ped.bin'))

    plt.close('all')


if __name__ == "__main__":
    output_dir = 'temp'

    if os.path.isdir(output_dir):
        rmtree(output_dir)

    os.mkdir(output_dir)

    generate_random_population(output_dir)
