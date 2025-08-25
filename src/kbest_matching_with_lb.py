import networkx as nx
from networkx.algorithms import bipartite, shortest_paths
import torch
import dgl


class GedLowerBound(object):
    def __init__(self, g1, g2, lb_setting=0):
        self.g1 = g1
        self.g2 = g2
        self.lb_setting = lb_setting
        self.n1 = g1.num_nodes()
        self.n2 = g2.num_nodes()
        assert self.n1 <= self.n2
        if g1.ndata['f'].shape[1] == 1:
            self.has_node_label = False
        else:
            self.has_node_label = True

    @staticmethod
    def mc(sg1, sg2):
        A = (sg1.adj() - sg2.adj()).coalesce()
        A_ged = (A ** 2).sum().item()
        F = sg1.ndata['f'] - sg2.ndata['f']
        F_ged = (F ** 2).sum().item()
        return (A_ged + F_ged) / 2.0

    def label_set(self, left_nodes, right_nodes):
        if right_nodes is None:
            return None

        partial_n = len(left_nodes)
        if partial_n == 0 and len(right_nodes) == self.n1:
            left_nodes = list(range(self.n1))
            partial_n = self.n1
        assert partial_n == len(right_nodes) and partial_n <= self.n1

        sub_g1 = self.g1.subgraph(left_nodes)
        sub_g2 = self.g2.subgraph(right_nodes)
        lb = self.mc(sub_g1, sub_g2)

        m1 = self.g1.num_edges() - self.n1 - sub_g1.num_edges()
        m2 = self.g2.num_edges() - self.n2 - sub_g2.num_edges()
        lb += abs(m1 - m2) / 2.0

        if (not self.has_node_label) or (partial_n == self.n1):
            lb += (self.n2 - self.n1)
        else:
            f1 = dgl.remove_nodes(self.g1, left_nodes).ndata['f'].sum(dim=0)
            f2 = dgl.remove_nodes(self.g2, right_nodes).ndata['f'].sum(dim=0)
            intersect = torch.min(f1, f2)
            lb += (max(f1.sum().item(), f2.sum().item()) - intersect.sum().item())

        return lb


class Subspace(object):
    def __init__(self, G, matching, res, I=None, O=None):
        self.G = G
        self.best_matching = matching
        self.best_res = res
        self.I = set() if I is None else I
        self.O = [] if O is None else O
        self.get_second_matching()
        self.lb = None 
        self.ged = None
        self.ged2 = None

    def __repr__(self):
        best_res = "1st matching: {} {}".format(self.best_matching, self.best_res)
        second_res = "2nd matching: {} {}".format(self.second_matching, self.second_res)
        IO = "I: {}\tO: {}\tbranch edge: {}".format(self.I, self.O, self.branch_edge)
        return best_res + "\n" + second_res + "\n" + IO

    def get_second_matching(self):
        G = self.G.copy()
        matching = self.best_matching.copy()
        n1 = len(matching)
        n = G.number_of_nodes()
        n2 = n - n1

        for (u, v) in self.O:
            G[u][v]["weight"] = float("inf")

        matched = [False] * n2
        for u in range(n1):
            v = matching[u]
            matched[v] = True
            v += n1
            w = -G[u][v]["weight"]
            if u in self.I:
                w = float("inf")
            G.remove_edge(u, v)
            G.add_edge(v, u, weight=w)
        G.add_node(n, bipartite=0)
        for v in range(n2):
            if matched[v]:
                G.add_edge(n, n1 + v, weight=0.0)
            else:
                G.add_edge(n1 + v, n, weight=0.0)

        dis = shortest_paths.dense.floyd_warshall(G)
        cycle_min_weight = float("inf")
        cycle_min_uv = None
        for u in range(n1):
            if u in self.I:
                continue
            v = matching[u] + n1
            res = dis[u][v] + G[v][u]["weight"]
            if res < cycle_min_weight:
                cycle_min_weight = res
                cycle_min_uv = (u, v)

        if cycle_min_uv is None:
            self.second_matching = None
            self.second_res = None
            self.branch_edge = None
            return

        u, v = cycle_min_uv
        length, path = shortest_paths.weighted.single_source_bellman_ford(G, source=u, target=v)
        assert abs(length + G[v][u]["weight"] - cycle_min_weight) < 1e-12
        self.branch_edge = (u, v)
        for i in range(0, len(path), 2):
            u, v = path[i], path[i + 1] - n1
            if u != n:
                matching[u] = v
        self.second_matching = matching
        self.second_res = self.best_res - cycle_min_weight

    def split(self):
        u, v = self.branch_edge

        I = self.I.copy()
        self.I.add(u)
        O = self.O.copy()
        O.append((u, v))

        G = self.G
        second_matching = self.second_matching
        self.second_matching = None
        second_res = self.second_res
        self.second_res = None

        self.get_second_matching()
        sp_new = Subspace(G, second_matching, second_res, I, O)
        return sp_new


class KBestMSolver(object):
    def __init__(self, a, g1, g2, pre_ged=None):
        G, best_matching, res = self.from_tensor_to_nx(a)
        sp = Subspace(G, best_matching, res)

        self.lb = GedLowerBound(g1, g2)
        self.lb_value = sp.lb = self.lb.label_set([], [])
        sp.ged = self.lb.label_set([], sp.best_matching)
        self.min_ged = sp.ged
        sp.ged2 = self.lb.label_set([], sp.second_matching)
        self.set_min_ged(sp.ged2)

        self.subspaces = [sp]
        self.k = 1
        self.expandable = True

        self.pre_ged = pre_ged

    def set_min_ged(self, ged):
        if ged is None:
            return
        if ged < self.min_ged:
            self.min_ged = ged

    @staticmethod
    def from_tensor_to_nx(A):
        n1, n2 = A.shape
        assert n1 <= n2
        top_nodes = range(n1)
        bottom_nodes = range(n1, n1 + n2)

        G = nx.DiGraph()
        G.add_nodes_from(top_nodes, bipartite=0)
        G.add_nodes_from(bottom_nodes, bipartite=1)
        A = A.tolist()
        for u in top_nodes:
            for v in bottom_nodes:
                G.add_edge(u, v, weight=-A[u][v - n1])

        matching = bipartite.matching.minimum_weight_full_matching(G, top_nodes)
        matching = [matching[u] - n1 for u in top_nodes]
        res = 0 
        for u in top_nodes:
            v = matching[u]
            res += A[u][v]
        return G, matching, res

    def expand_subspaces(self):

        max_res = -1
        max_spid = None

        for spid, sp in enumerate(self.subspaces):
            if sp.lb < self.min_ged and sp.second_res is not None and sp.second_res > max_res:
                max_res = sp.second_res
                max_spid = spid

        if max_spid is None:
            self.expandable = False
            return

        sp = self.subspaces[max_spid]
        sp_new = sp.split()
        self.subspaces.append(sp_new)
        self.k += 1

        sp_new.lb = sp.lb
        sp_new.ged = sp.ged2
        sp_new.ged2 = self.lb.label_set([], sp_new.second_matching)
        self.set_min_ged(sp_new.ged2)

        left_nodes = list(sp.I)
        right_nodes = [sp.best_matching[u] for u in left_nodes]
        sp.lb = self.lb.label_set(left_nodes, right_nodes)
        sp.ged2 = self.lb.label_set([], sp.second_matching)
        self.set_min_ged(sp.ged2)

    def get_matching(self, k):
        while self.k < k and self.expandable:
            self.expand_subspaces()
        if self.k < k:
            return None, None, None
        else:
            sp = self.subspaces[k-1]
            return sp.best_matching, sp.best_res, sp.ged

    def best_matching(self):
        for sp in self.subspaces:
            if sp.ged == self.min_ged:
                return sp.best_matching
            elif sp.ged2 == self.min_ged:
                return sp.second_matching
        print("GED Error: no sp's ged or ged2 = self.min_ged")
        return None

