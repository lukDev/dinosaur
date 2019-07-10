import math_helper
import random


class Gene:
    def __init__(self, in_n, out_n, weight, enabled, innovation_number):
        self.in_n = in_n
        self.out_n = out_n
        self.weight = weight
        self.enabled = enabled
        self.innovation_number = innovation_number

    def replicate(self):
        return Gene(self.in_n, self.out_n, self.weight, self.enabled, self.innovation_number)


class Network:
    def __init__(self, genes, add_ins, add_outs):
        self.genes = genes
        self.ins = []
        self.outs = []
        self.all_neurons = []
        self.add_ins = add_ins
        self.add_outs = add_outs
        self.set_ins_and_outs()

    def set_ins_and_outs(self):
        in_counts = dict()
        out_counts = dict()

        for gene in self.genes:
            if gene.in_n in out_counts.keys():
                out_counts[gene.in_n] += 1
            else:
                out_counts[gene.in_n] = 1

            if gene.in_n not in in_counts.keys():
                in_counts[gene.in_n] = 0

            if gene.out_n in in_counts.keys():
                in_counts[gene.out_n] += 1
            else:
                in_counts[gene.out_n] = 1

            if gene.out_n not in out_counts.keys():
                out_counts[gene.out_n] = 0
                
        self.ins = [n for (n, ic) in in_counts.items() if ic == 0 and out_counts[n] != 0]
        if -1 not in self.ins:
            self.ins.insert(0, -1)  # bias
        for add_in in self.add_ins:
            if add_in not in self.ins:
                self.ins.append(add_in)

        self.outs = [n for (n, oc) in out_counts.items() if oc == 0 and in_counts[n] != 0]
        for add_out in self.add_outs:
            if add_out not in self.outs:
                self.outs.append(add_out)

        self.all_neurons = [n for (n, _) in in_counts.items()]
        if -1 not in self.all_neurons:
            self.all_neurons.insert(0, -1)  # bias
        for add_in in self.add_ins:
            if add_in not in self.all_neurons:
                self.all_neurons.append(add_in)
        for add_out in self.add_outs:
            if add_out not in self.all_neurons:
                self.all_neurons.append(add_out)

    def refresh_neuron_list(self):
        neurons = []

        for gene in self.genes:
            if gene.in_n not in neurons:
                neurons.append(gene.in_n)
            if gene.out_n not in neurons:
                neurons.append(gene.out_n)

        neurons.insert(0, -1)  # bias

        self.all_neurons = neurons

    def add_activation_for(self, neuron_nr, activations):
        if neuron_nr in activations.keys():
            return activations

        activation = 0
        for gene in self.genes:
            if not gene.enabled:
                continue

            if gene.out_n == neuron_nr:
                n_activations = self.add_activation_for(gene.in_n, activations)
                activation += gene.weight * n_activations[gene.in_n]
        activations[neuron_nr] = math_helper.sigmoid_activation(activation)

        return activations

    def get_activations(self, in_acts):
        if len(in_acts) != len(self.ins):
            raise ValueError("Input activations don't match input neurons")

        activations = {self.ins[i] : in_acts[i] for i in range(len(in_acts))}
        for out in self.outs:
            activations = self.add_activation_for(out, activations)

        return [ac for (n, ac) in activations.items() if n in self.outs]

    def replicate(self):
        return Network([g.replicate() for g in self.genes], self.add_ins, self.add_outs)

    def new_gene_neurons(self):
        in_n = random.choice(self.all_neurons)
        out_n = random.choice(self.all_neurons)

        if in_n == out_n or in_n in self.outs or out_n in self.ins:
            return self.new_gene_neurons()

        if self._is_dependency_(out_n, [in_n]):
            return self.new_gene_neurons()

        return (in_n, out_n)

    def _is_dependency_(self, neuron, path):
        if len(path) == 0:
            return False

        if path[0] == neuron:
            return True
        else:
            path.extend([g.in_n for g in self.genes if g.out_n == path[0]])
            return self._is_dependency_(neuron, path[1:])

    def max_innovation_number(self):
        return max([g.innovation_number for g in self.genes]) if len(self.genes) > 0 else 0

    def to_string(self):
        string = "Network:\n"

        for gene in self.genes:
            string += "\t%d: [%d (%+.3f->) %d] (%s)\n" % (gene.innovation_number, gene.in_n, gene.weight, gene.out_n, "e" if gene.enabled else "d")

        return string
