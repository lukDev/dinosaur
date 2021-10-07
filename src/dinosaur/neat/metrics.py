from dinosaur.utils.data import Data
import random
from dinosaur.neat.network import Network
from dinosaur.neat.network import Gene
from dinosaur.utils import math_helper


# --- Fitness Evaluation ---

# returns the player with the highest fitness
def best_player():
    fitnesses = [p.fitness for p in Data.players]
    max_f = max(fitnesses)
    max_is = [i for i in range(len(fitnesses)) if fitnesses[i] == max_f]

    if len(max_is) == 0:
        return None
    else:
        return Data.players[max_is[0]]


# returns the average fitness of a given species
def avg_fitness(specy):
    fit_sum = 0.
    for (_, f) in specy:
        fit_sum += f

    return fit_sum / len(specy)


# returns the average fitness of all species
def total_avg_fitness(species):
    sum = 0.
    for specy in species:
        sum += avg_fitness(specy)

    return sum


# returns the desired amount of children to be created from a certain species, depending on the average fitness of
# all species
def offspring_count(specy, total_avg_fit):
    return int(avg_fitness(specy) / total_avg_fit * Data.population_size)


# --- Network Differences ---

# for given networks n1 and n2, returns the excess genes of either n1 or n2 in relation to the other,
# depending on which network has the higher maximum innovation number in its genes, and indicates which one that is
def excess_genes(n1, n2):
    if len(n1.genes) == 0:
        return n2.genes, False
    elif len(n2.genes) == 0:
        return n1.genes, True

    max_inno_nr_1 = max([g.innovation_number for g in n1.genes])
    max_inno_nr_2 = max([g.innovation_number for g in n2.genes])

    if max_inno_nr_1 < max_inno_nr_2:
        first_has_excess = False
        ex_genes = [g for g in n2.genes if g.innovation_number > max_inno_nr_1]
    else:
        first_has_excess = True
        ex_genes = [g for g in n1.genes if g.innovation_number > max_inno_nr_2]

    return ex_genes, first_has_excess


# returns the genes of given networks n1 and n2 that don't match the other one's
def disjoint_genes(n1, n2):
    if len(n1.genes) == 0 or len(n2.genes) == 0:
        return [], []

    innos_1 = [g.innovation_number for g in n1.genes]
    innos_2 = [g.innovation_number for g in n2.genes]
    limit = min(max(innos_1), max(innos_2))

    dis_genes_1 = [g for g in n1.genes if g.innovation_number <= limit and g.innovation_number not in innos_2]
    dis_genes_2 = [g for g in n2.genes if g.innovation_number <= limit and g.innovation_number not in innos_1]

    return dis_genes_1, dis_genes_2


# finds all pairs of matching genes across two networks n1 and n2 and for each pair randomly selects one of the two
# returns the result
def random_matching_genes(n1, n2):
    matching_genes = []

    for g1 in n1.genes:
        m_genes = [g2 for g2 in n2.genes if g1.innovation_number == g2.innovation_number]
        if len(m_genes) > 0:
            random_i = random.randint(0, 1)
            matching_genes.append([g1, m_genes[0]][random_i])

    return matching_genes


# returns the difference between two networks based on their excess and disjoint genes and their average weights
def network_dif(n1, n2):
    max_size = float(max(len(n1.genes), len(n2.genes)))
    if max_size == 0:
        return 0.

    (ex_genes, _) = excess_genes(n1, n2)
    (d_genes_1, d_genes_2) = disjoint_genes(n1, n2)

    return Data.excess_weight * len(ex_genes) / max_size + Data.disjoint_weight * (len(d_genes_1)
                    + len(d_genes_2)) / max_size + Data.weight_dif_weight * abs((n1.avg_weight() - n2.avg_weight()))


# --- Mutation ---

# performs NEAT evolution on a given set of networks and returns the result
def evolve(nets):
    if len(nets) == 0:
        print("No networks passed for evolving. Is this intentional?")
        return nets

    # speciation
    species = []
    for (net, fitness) in nets:
        for specy in species:
            (rep, _) = specy[0]  # should just fail bluntly if there's no networks in the species
            if network_dif(net, rep) <= Data.speciation_threshold:
                specy.append((net, fitness))
                break
        else:
            species.append([(net, fitness)])

    # cut off the worst half of all species
    new_species = []
    for specy in species:
        cap_index = (len(specy) + 1) // 2
        new_specy = sorted(specy, key=lambda af: af[1], reverse=True)[:cap_index]
        new_species.append(new_specy)
    species = new_species

    # remove weak species
    new_species = []
    total_avg_fit = total_avg_fitness(species)
    for specy in species:
        if offspring_count(specy, total_avg_fit) >= 1:
            new_species.append(specy)
    species = new_species

    # breed within each species
    children = []
    total_avg_fit = total_avg_fitness(species)
    for specy in species:
        off_count = offspring_count(specy, total_avg_fit) - 1
        for _ in range(off_count):
            children.append(breed(specy))

    # randomly add the best of each species until there are enough
    while len(children) < Data.population_size - len(species):
        (c, _) = random.choice(species)[0]
        children.append(c)

    # mutate and add unadulterated parents
    new_population = mutate(children)
    for specy in species:
        (c, _) = specy[0]
        new_population.append(c)

    return new_population


# performs mutation on a given set of networks and keeps track of the innovation numbers used and created in the process
def mutate(nets):
    add_gene_mutations = []  # (in neuron, out neuron, innovation nr)
    add_neuron_mutations = []  # (in neuron, out neuron, innovation nr in, innovation nr out)

    for net in nets:
        for gene in net.genes:
            if math_helper.random_decision(Data.prob_weight_mut):
                if math_helper.random_decision(Data.prob_weight_nudged):
                    gene.weight = math_helper.sigmoid_weight(gene.weight
                                                             + random.uniform(-Data.weight_nudge_limit, Data.weight_nudge_limit))
                else:
                    gene.weight = random.uniform(-1., 1.)

        add_neuron = math_helper.random_decision(Data.prob_new_neuron)
        add_gene = math_helper.random_decision(Data.prob_new_gene)

        if add_gene:
            if not add_neuron:
                (in_n, out_n) = net.new_gene_neurons()
                weight = random.uniform(-1., 1.)

                prev_similar_innovations = [p_inno_nr for (p_in_n, p_out_n, p_inno_nr) in add_gene_mutations
                                            if p_in_n == in_n and p_out_n == out_n]

                if len(prev_similar_innovations) > 0:
                    innovation_number = prev_similar_innovations[0]
                else:
                    Data.current_innovation_number += 1
                    innovation_number = Data.current_innovation_number
                    add_gene_mutations.append((in_n, out_n, innovation_number))

                net.genes.append(Gene(in_n=in_n, out_n=out_n, weight=weight, enabled=True,
                                      innovation_number=innovation_number))
        else:
            if add_neuron and len(net.genes) > 0:
                split_gene = random.choice(net.genes)
                split_gene.enabled = False
                new_neuron_number = max(net.all_neurons) + 1

                prev_similar_innovations = [(p_inno_nr_in, p_inno_nr_out)
                                            for (p_in_n, p_out_n, p_inno_nr_in, p_inno_nr_out) in add_neuron_mutations
                                            if p_in_n == split_gene.in_n and p_out_n == split_gene.out_n]

                if len(prev_similar_innovations) > 0:
                    (innovation_number_in, innovation_number_out) = prev_similar_innovations[0]
                else:
                    Data.current_innovation_number += 1
                    innovation_number_in = Data.current_innovation_number
                    Data.current_innovation_number += 1
                    innovation_number_out = Data.current_innovation_number

                net.genes.append(Gene(in_n=split_gene.in_n, out_n=new_neuron_number, weight=random.uniform(-1., 1.),
                                      enabled=True, innovation_number=innovation_number_in))
                net.genes.append(Gene(in_n=new_neuron_number, out_n=split_gene.out_n, weight=random.uniform(-1., 1.),
                                      enabled=True, innovation_number=innovation_number_out))

                net.refresh_neuron_list()

    return nets


# --- Reproduction ---

# creates a child from within a species, either through crossover or through simple replication
def breed(specy):
    if math_helper.random_decision(Data.prob_crossover):
        (n1, _) = random.choice(specy)
        (n2, _) = random.choice(specy)
        return produce_child(n1, n2)
    else:
        (n, _) = random.choice(specy)
        return n.replicate()


# creates a child from two given networks
def produce_child(n1, n2):
    child_genes = random_matching_genes(n1, n2)
    (dis_genes_1, _) = disjoint_genes(n1, n2)
    (ex_genes, first_has_excess) = excess_genes(n1, n2)

    child_genes.extend(dis_genes_1)
    if first_has_excess:
        child_genes.extend(ex_genes)

    return Network(child_genes, n1.add_ins, n1.add_outs)
