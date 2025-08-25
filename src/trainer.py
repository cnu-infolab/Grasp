import sys
import time
import dgl
import torch
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
from utils import load_all_graphs, load_labels, load_ged
import matplotlib.pyplot as plt
from kbest_matching_with_lb import KBestMSolver
from math import exp
from scipy.stats import spearmanr, kendalltau
import networkx as nx
from models import Grasp
from GedMatrix import fixed_mapping_loss
from sklearn.metrics import roc_auc_score


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.load_data_time = 0.0
        self.to_torch_time = 0.0
        self.results = []
        
        self.seed = 1
        print("seed : ", self.seed)
        self.seed_everything(self.seed)
        self.use_gpu = False
        print("use_gpu =", self.use_gpu)
        self.device = torch.device('cuda') if self.use_gpu else torch.device('cpu')

        self.load_data()
        self.transfer_data_to_torch()
        self.delta_graphs = [None] * len(self.graphs)
        self.gen_delta_graphs()
        self.init_graph_pairs()
        self.setup_model()
        
    def seed_everything(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    def setup_model(self):
        if self.args.model_name == "Grasp":
            self.args.loss_weight = 8.0
            self.args.gtmap = True
            self.model = Grasp(self.args, self.number_of_labels).to(self.device)
        else:
            assert False

    def process_batch(self, batch):
        self.optimizer.zero_grad()
        losses = torch.tensor([0]).float().to(self.device)
        if self.args.model_name == "Grasp":
            weight = self.args.loss_weight
            for graph_pair in batch:
                data = self.pack_graph_pair(graph_pair)
                target, gt_mapping = data["target"], data["mapping"]
                prediction, _, mapping = self.model(data)
                losses = losses + (10.0-weight)*fixed_mapping_loss(mapping, gt_mapping) + weight * F.mse_loss(target, prediction)
        else:
            assert False
            
        losses.backward()
        self.optimizer.step()
        return losses.item()

    def load_data(self):
        t1 = time.time()
        dataset_name = self.args.dataset
        self.train_num, self.val_num, self.test_num, self.graphs = load_all_graphs(self.args.abs_path, dataset_name)
        print("Load {} graphs. ({} for training)".format(len(self.graphs), self.train_num))

        self.number_of_labels = 0
        if dataset_name in ['AIDS']:
            self.global_labels, self.features = load_labels(self.args.abs_path, dataset_name)
            self.number_of_labels = len(self.global_labels)
        if self.number_of_labels == 0:
            self.number_of_labels = 1
            self.features = []
            for g in self.graphs:
                self.features.append([[2.0] for u in range(g['n'])])
        ged_dict = dict()
        load_ged(ged_dict, self.args.abs_path, dataset_name, 'TaGED.json')
        self.ged_dict = ged_dict
        print("Load ged dict.")
        t2 = time.time()
        self.load_data_time = t2 - t1

    def transfer_data_to_torch(self):
        t1 = time.time()

        self.edge_index = []
        for g in self.graphs:
            edge = g['graph']
            edge = edge + [[y, x] for x, y in edge]
            edge = edge + [[x, x] for x in range(g['n'])]
            edge = torch.tensor(edge).t().long().to(self.device)
            self.edge_index.append(edge)
        self.features = [torch.tensor(x).float().to(self.device) for x in self.features]
        print("Feature shape of 1st graph:", self.features[0].shape)

        n = len(self.graphs)
        mapping = [[None for i in range(n)] for j in range(n)]
        ged = [[(0., 0., 0., 0.) for i in range(n)] for j in range(n)]
        gid = [g['gid'] for g in self.graphs]
        self.gid = gid
        self.gn = [g['n'] for g in self.graphs]
        self.gm = [g['m'] for g in self.graphs]
        for i in range(n):
            mapping[i][i] = torch.eye(self.gn[i], dtype=torch.float, device=self.device)
            for j in range(i + 1, n):
                id_pair = (gid[i], gid[j])
                n1, n2 = self.gn[i], self.gn[j]
                if id_pair not in self.ged_dict:
                    id_pair = (gid[j], gid[i])
                    n1, n2 = n2, n1
                if id_pair not in self.ged_dict:
                    ged[i][j] = ged[j][i] = None
                    mapping[i][j] = mapping[j][i] = None
                else:
                    ta_ged, gt_mappings = self.ged_dict[id_pair]
                    ged[i][j] = ged[j][i] = ta_ged
                    mapping_list = [[0 for y in range(n2)] for x in range(n1)]
                    for gt_mapping in gt_mappings:
                        for x, y in enumerate(gt_mapping):
                            mapping_list[x][y] = 1
                    mapping_matrix = torch.tensor(mapping_list).float().to(self.device)
                    mapping[i][j] = mapping[j][i] = mapping_matrix
        self.ged = ged
        self.mapping = mapping

        t2 = time.time()
        self.to_torch_time = t2 - t1

    @staticmethod
    def delta_graph(g, f, device):
        new_data = dict()

        n = g['n']
        permute = list(range(n))
        random.shuffle(permute)
        mapping = torch.sparse_coo_tensor((list(range(n)), permute), [1.0] * n, (n, n)).to_dense().to(device)

        edge = g['graph']
        edge_set = set()
        for x, y in edge:
            edge_set.add((x, y))
            edge_set.add((y, x))

        random.shuffle(edge)
        m = len(edge)
        ged = random.randint(1, 5) if n <= 20 else random.randint(1, 10)
        del_num = min(m, random.randint(0, ged))
        edge = edge[:(m - del_num)]
        add_num = ged - del_num
        if (add_num + m) * 2 > n * (n - 1):
            add_num = n * (n - 1) // 2 - m
        cnt = 0
        while cnt < add_num:
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if (x != y) and (x, y) not in edge_set:
                edge_set.add((x, y))
                edge_set.add((y, x))
                cnt += 1
                edge.append([x, y])
        assert len(edge) == m - del_num + add_num
        new_data["n"] = n
        new_data["m"] = len(edge)

        new_edge = [[permute[x], permute[y]] for x, y in edge]
        new_edge = new_edge + [[y, x] for x, y in new_edge]
        new_edge = new_edge + [[x, x] for x in range(n)]

        new_edge = torch.tensor(new_edge).t().long().to(device)

        feature2 = torch.zeros(f.shape).to(device)
        for x, y in enumerate(permute):
            feature2[y] = f[x]

        new_data["permute"] = permute
        new_data["mapping"] = mapping
        ged = del_num + add_num
        new_data["ta_ged"] = (ged, 0, 0, ged)
        new_data["edge_index"] = new_edge
        new_data["features"] = feature2
        return new_data

    def gen_delta_graphs(self):
        k = self.args.num_delta_graphs
        for i, g in enumerate(self.graphs):
            if g['n'] <= 10:
                continue
            f = self.features[i]
            self.delta_graphs[i] = [self.delta_graph(g, f, self.device) for j in range(k)]

    def check_pair(self, i, j):
        if i == j:
            return 0, i, j
        id1, id2 = self.gid[i], self.gid[j]
        if (id1, id2) in self.ged_dict:
            return 0, i, j
        elif (id2, id1) in self.ged_dict:
            return 0, j, i
        else:
            return None

    def init_graph_pairs(self):
        self.training_graphs = []
        self.val_graphs = []
        self.testing_graphs = []
        self.testing_graphs_small = []
        self.testing_graphs_large = []
        self.testing2_graphs = []

        train_num = self.train_num
        val_num = train_num + self.val_num
        test_num = len(self.graphs)

        if self.args.demo:
            train_num = 30
            val_num = 40
            test_num = 50
            self.args.epochs = 1

        assert self.args.graph_pair_mode == "combine"
        dg = self.delta_graphs
        for i in range(train_num):
            if self.gn[i] <= 10:
                for j in range(i, train_num):
                    tmp = self.check_pair(i, j)
                    if tmp is not None:
                        self.training_graphs.append(tmp)
            elif dg[i] is not None:
                k = len(dg[i])
                for j in range(k):
                    self.training_graphs.append((1, i, j))

        li = []
        for i in range(train_num):
            if self.gn[i] <= 10:
                li.append(i)
        print("The number of small training graphs:", len(li))

        for i in range(train_num, val_num):
            if self.gn[i] <= 10:
                random.shuffle(li)
                self.val_graphs.append((0, i, li[:self.args.num_testing_graphs]))
            elif dg[i] is not None:
                k = len(dg[i])
                self.val_graphs.append((1, i, list(range(k))))

        for i in range(val_num, test_num):
            if self.gn[i] <= 10:
                random.shuffle(li)
                self.testing_graphs.append((0, i, li[:self.args.num_testing_graphs]))
                self.testing_graphs_small.append((0, i, li[:self.args.num_testing_graphs]))
            elif dg[i] is not None:
                k = len(dg[i])
                self.testing_graphs.append((1, i, list(range(k))))
                self.testing_graphs_large.append((1, i, list(range(k))))

        li = []
        for i in range(val_num, test_num):
            if self.gn[i] <= 10:
                li.append(i)
        print("The number of small testing graphs:", len(li))

        for i in range(val_num, test_num):
            if self.gn[i] <= 10:
                random.shuffle(li)
                self.testing2_graphs.append((0, i, li[:self.args.num_testing_graphs]))
            elif dg[i] is not None:
                k = len(dg[i])
                self.testing2_graphs.append((1, i, list(range(k))))

        print("Generate {} training graph pairs.".format(len(self.training_graphs)))
        print("Generate {} * {} val graph pairs.".format(len(self.val_graphs), self.args.num_testing_graphs))
        print("Generate {} * {} testing graph pairs.".format(len(self.testing_graphs), self.args.num_testing_graphs))
        print("Generate {} * {} small testing graph pairs.".format(len(self.testing_graphs_small), self.args.num_testing_graphs))
        print("Generate {} * {} large testing graph pairs.".format(len(self.testing_graphs_large), self.args.num_testing_graphs))
        print("Generate {} * {} testing2 graph pairs.".format(len(self.testing2_graphs), self.args.num_testing_graphs))

    def create_batches(self):
        random.shuffle(self.training_graphs)
        batches = []
        for graph in range(0, len(self.training_graphs), self.args.batch_size):
            batches.append(self.training_graphs[graph:graph + self.args.batch_size])
        return batches

    def pack_graph_pair(self, graph_pair):
        new_data = dict()

        (pair_type, id_1, id_2) = graph_pair
        if pair_type == 0:
            gid_pair = (self.gid[id_1], self.gid[id_2])
            if gid_pair not in self.ged_dict:
                id_1, id_2 = (id_2, id_1)
                gid_pair = (self.gid[id_1], self.gid[id_2])

            real_ged = self.ged[id_1][id_2][0]
            ta_ged = self.ged[id_1][id_2][1:]

            new_data["id_1"] = id_1
            new_data["id_2"] = id_2

            new_data["edge_index_1"] = self.edge_index[id_1]
            new_data["edge_index_2"] = self.edge_index[id_2]
            new_data["features_1"] = self.features[id_1]
            new_data["features_2"] = self.features[id_2]

            if self.args.gtmap:
                new_data["mapping"] = self.mapping[id_1][id_2]

            new_data["permute"] = [list(range(self.gn[id_1]))] if id_1 == id_2 else self.ged_dict[gid_pair][1]

        elif pair_type == 1:
            new_data["id"] = id_1
            dg: dict = self.delta_graphs[id_1][id_2]

            real_ged = dg["ta_ged"][0]
            ta_ged = dg["ta_ged"][1:]

            new_data["edge_index_1"] = self.edge_index[id_1]
            new_data["edge_index_2"] = dg["edge_index"]
            new_data["features_1"] = self.features[id_1]
            new_data["features_2"] = dg["features"]

            new_data["permute"] = [dg["permute"]]
            if self.args.gtmap:
                new_data["mapping"] = dg["mapping"]
        else:
            assert False

        n1, m1 = (self.gn[id_1], self.gm[id_1])
        n2, m2 = (self.gn[id_2], self.gm[id_2]) if pair_type == 0 else (dg["n"], dg["m"])
        new_data["n1"] = n1
        new_data["n2"] = n2
        new_data["ged"] = real_ged
        if self.args.target_mode == "exp":
            avg_v = (n1 + n2) / 2.0
            new_data["avg_v"] = avg_v
            new_data["target"] = torch.exp(torch.tensor([-real_ged / avg_v]).float()).to(self.device)
            new_data["ta_ged"] = torch.exp(torch.tensor(ta_ged).float() / -avg_v).to(self.device)
        elif self.args.target_mode == "linear":
            higher_bound = max(n1, n2) + max(m1, m2)
            new_data["hb"] = higher_bound
            new_data["target"] = torch.tensor([real_ged / higher_bound]).float().to(self.device)
            new_data["ta_ged"] = (torch.tensor(ta_ged).float() / higher_bound).to(self.device)
        else:
            assert False

        return new_data

    def fit(self):
        print("\nModel training.\n")
        t1 = time.time()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)

        self.model.train()
        self.values = []
        with tqdm(total=self.args.epochs * len(self.training_graphs), unit="graph_pairs", leave=True, desc="Epoch",
                  file=sys.stdout) as pbar:
            for epoch in range(self.args.epochs):
                batches = self.create_batches()
                loss_sum = 0
                main_index = 0
                for index, batch in enumerate(batches):
                    batch_total_loss = self.process_batch(batch)
                    loss_sum += batch_total_loss
                    main_index += len(batch)
                    loss = loss_sum / main_index
                    pbar.update(len(batch))
                    pbar.set_description(
                        "Epoch_{}: loss={} - Batch_{}: loss={}".format(self.cur_epoch + 1, round(1000 * loss, 3),
                                                                       index,
                                                                       round(1000 * batch_total_loss / len(batch), 3)))
                tqdm.write("Epoch {}: loss={}".format(self.cur_epoch + 1, round(1000 * loss, 3)))
                training_loss = round(1000 * loss, 3)
        t2 = time.time()
        training_time = t2 - t1

        self.results.append(
            ('model_name', 'dataset', "graph_set", "current_epoch", "training_time(s/epoch)", "training_loss(1000x)"))
        self.results.append(
            (self.args.model_name, self.args.dataset, "train", self.cur_epoch + 1, training_time, training_loss))

        print(*self.results[-2], sep='\t')
        print(*self.results[-1], sep='\t')
        with open(self.args.abs_path + self.args.result_path+self.args.dataset+'/results_'+self.args.model_name+'.txt', 'a') as f:
            print("## Training", file=f)
            print("```", file=f)
            print(*self.results[-2], sep='\t', file=f)
            print(*self.results[-1], sep='\t', file=f)
            print("```\n", file=f)

    @staticmethod
    def cal_pk(num, pre, gt):
        if num >= len(pre):
            return -1.0
        tmp = list(zip(gt, pre))
        tmp.sort()
        beta = []
        for i, p in enumerate(tmp):
            beta.append((p[1], p[0], i))
        beta.sort()
        ans = 0
        for i in range(num):
            if beta[i][2] < num:
                ans += 1
        return ans / num

    def gen_edit_path(self, data, permute):
        n1, n2 = data["n1"], data["n2"]
        raw_edges_1, raw_edges_2 = data["edge_index_1"].t().tolist(), data["edge_index_2"].t().tolist()
        raw_f1, raw_f2 = data["features_1"].tolist(), data["features_2"].tolist()
        assert len(permute) == n1
        assert len(raw_f1) == n1 and len(raw_f2) == n2 and len(raw_f1[0]) == len(raw_f2[0])

        edges_1 = set()
        for (u, v) in raw_edges_1:
            pu, pv = permute[u], permute[v]
            if pu <= pv:
                edges_1.add((pu, pv))

        edges_2 = set()
        for (u, v) in raw_edges_2:
            if u <= v:
                edges_2.add((u, v))

        edit_edges = edges_1 ^ edges_2

        f1 = []
        num_label = len(raw_f1[0])
        for f in raw_f1:
            for j in range(num_label):
                if f[j] > 0:
                    f1.append(j)
                    break
        f2 = []
        for f in raw_f2:
            for j in range(num_label):
                if f[j] > 0:
                    f2.append(j)
                    break

        relabel_nodes = set()
        for (u, v) in enumerate(permute):
            if f1[u] != f2[v]:
                relabel_nodes.add((v, f1[u]))

        return edit_edges, relabel_nodes

    def score_my(self, testing_graph_set='test',test_k=None):
        print("\n\nModel evaluation on {} set.\n".format(testing_graph_set))
        if testing_graph_set == 'test':
            testing_graphs = self.testing_graphs
        elif testing_graph_set == 'test_small':
            testing_graphs = self.testing_graphs_small
        elif testing_graph_set == 'test_large':
            testing_graphs = self.testing_graphs_large
        elif testing_graph_set == 'test2':
            testing_graphs = self.testing2_graphs
        elif testing_graph_set == 'val':
            testing_graphs = self.val_graphs
        else:
            assert False

        num = 0
        time_usage = []
        mse = []
        mae = []
        num_acc = 0
        num_fea = 0
        rho = []
        tau = []
        pk10 = []
        pk20 = []

        for pair_type, i, j_list in tqdm(testing_graphs, file=sys.stdout):
            pre = []
            gt = []
            t1 = time.time()
            for j in j_list:
                data = self.pack_graph_pair((pair_type, i, j))
                target, gt_ged = data["target"].item(), data["ged"]
                if test_k == None:
                    model_out = self.model(data)
                elif test_k == 0:
                    model_out = self.test_noah(data)
                prediction, pre_ged = model_out[0], model_out[1]
                if self.args.GW:
                    gw = GEDGW(data, self.args)
                    out1 =  gw.process()
                    pre_ged2 = out1[1]
                    pre_ged = min(pre_ged,pre_ged2)
                round_pre_ged = round(pre_ged)

                num += 1
                if prediction is None:
                    mse.append(-0.001)
                elif prediction.shape[0] == 1:
                    mse.append((prediction.item() - target) ** 2)
                else:
                    mse.append(F.mse_loss(prediction, data["ta_ged"]).item())
                pre.append(pre_ged)
                gt.append(gt_ged)

                mae.append(abs(round_pre_ged - gt_ged))
                if round_pre_ged == gt_ged:
                    num_acc += 1
                    num_fea += 1
                elif round_pre_ged > gt_ged:
                    num_fea += 1
            t2 = time.time()
            time_usage.append(t2 - t1)

            rho.append(spearmanr(pre, gt)[0])
            tau.append(kendalltau(pre, gt)[0])
            if rho[-1] != rho[-1]:
                rho[-1] = 0.
            if tau[-1] != tau[-1]:
                tau[-1] = 0.
            pk10.append(self.cal_pk(10, pre, gt))
            pk20.append(self.cal_pk(20, pre, gt))

        time_usage = round(np.mean(time_usage), 3)
        mse = round(np.mean(mse) * 1000, 3)
        mae = round(np.mean(mae), 3)
        acc = round(num_acc / num, 3)
        fea = round(num_fea / num, 3)

        rho = round(np.mean(rho), 3)
        tau = round(np.mean(tau), 3)
        pk10 = round(np.mean(pk10), 3)
        pk20 = round(np.mean(pk20), 3)

        self.results.append(
            ('model_name', 'dataset', 'graph_set', '#testing_pairs', 'time_usage(s/pair)', 'mse', 'mae', 'acc',
             'fea', 'rho', 'tau', 'pk10', 'pk20'))
        self.results.append((self.args.model_name, self.args.dataset, testing_graph_set, num, time_usage, mse, mae, acc,
                             fea, rho, tau, pk10, pk20))

        print(*self.results[-2], sep='\t')
        print(*self.results[-1], sep='\t')
        with open(self.args.abs_path + self.args.result_path+self.args.dataset+'/results_'+self.args.model_name+'.txt', 'a') as f:
            print("## Testing", file=f)
            print("```", file=f)
            print(*self.results[-2], sep='\t', file=f)
            print(*self.results[-1], sep='\t', file=f)
            print("```\n", file=f)
        
    def path_score_my(self, testing_graph_set='test', test_k=None):
        print("\n\nModel evaluation on {} set.\n".format(testing_graph_set))
        if testing_graph_set == 'test':
            testing_graphs = self.testing_graphs
        elif testing_graph_set == 'test_small':
            testing_graphs = self.testing_graphs_small
        elif testing_graph_set == 'test_large':
            testing_graphs = self.testing_graphs_large
        elif testing_graph_set == 'test2':
            testing_graphs = self.testing2_graphs
        elif testing_graph_set == 'val':
            testing_graphs = self.val_graphs
        else:
            assert False

        num = 0
        time_usage = []

        mae_path = []

        num_acc = 0
        num_fea = 0
        rho = []
        tau = []
        pk10 = []
        pk20 = []
        rate = []
        recall = []
        precision = []
        f1 = []
        sim = []

        for pair_type, i, j_list in tqdm(testing_graphs[10:], file=sys.stdout):
            pre = []
            gt = []
            t1 = time.time()
            for j in j_list:
                data = self.pack_graph_pair((pair_type, i, j))
                target, gt_ged = data["target"].item(), data["ged"]
                if gt_ged == 0:
                    continue
                if test_k is None:
                    model_out = self.model(data)
                    prediction, pre_ged = model_out[0], model_out[1]
                elif test_k == 0:
                    model_out = self.test_noah(data)
                    pre_permute = model_out[2]
                    pre_edit_edges, pre_relabel_nodes = self.gen_edit_path(data, pre_permute)
                    prediction, pre_ged = model_out[0], model_out[1]
                    pre_ged = len(pre_edit_edges) + len(pre_relabel_nodes)
                elif test_k > 0:
                    model_out = self.test_matching(data, test_k=test_k)
                    pre_permute = model_out[2]
                    pre_edit_edges, pre_relabel_nodes = self.gen_edit_path(data, pre_permute)
                    prediction, pre_ged = model_out[0], model_out[1]
                else:
                    assert False

                round_pre_ged = round(pre_ged)

                num += 1
                pre.append(pre_ged)
                gt.append(gt_ged)

                mae_path.append(abs(round_pre_ged - gt_ged))
                if round_pre_ged== gt_ged:
                    num_acc += 1
                    num_fea += 1
                elif round_pre_ged > gt_ged:
                    num_fea += 1
                assert len(pre_edit_edges) + len(pre_relabel_nodes) == round_pre_ged

                best_rate = 0.
                best_recall = 0.
                best_precision = 0.
                best_f1 = 0.
                best_sim = 0.

                for permute in data["permute"]:
                    tmp = 0
                    for (v1, v2) in zip(permute, pre_permute):
                        if v1 == v2:
                            tmp += 1
                    best_rate = max(best_rate, tmp / data["n1"])

                    edit_edges, relabel_nodes = self.gen_edit_path(data, permute)
                    assert len(edit_edges) + len(relabel_nodes) == gt_ged
                    num_overlap = len(pre_edit_edges & edit_edges) + len(pre_relabel_nodes & relabel_nodes)

                    best_recall = max(best_recall, num_overlap / gt_ged)
                    best_precision = max(best_precision, num_overlap / round_pre_ged)
                    best_f1 = max(best_f1, 2.0 * num_overlap / (gt_ged + round_pre_ged))
                    best_sim = max(best_sim, num_overlap / (gt_ged + round_pre_ged - num_overlap))

                rate.append(best_rate)
                recall.append(best_recall)
                precision.append(best_precision)
                f1.append(best_f1)
                sim.append(best_sim)

            t2 = time.time()
            time_usage.append(t2 - t1)
            rho.append(spearmanr(pre, gt)[0])
            tau.append(kendalltau(pre, gt)[0])
            if rho[-1] != rho[-1]:
                rho[-1] = 0.
            if tau[-1] != tau[-1]:
                tau[-1] = 0.
            pk10.append(self.cal_pk(10, pre, gt))
            pk20.append(self.cal_pk(20, pre, gt))            

        time_usage = round(np.mean(time_usage), 3)
        mae_path = round(np.mean(mae_path), 3)
        acc = round(num_acc / num, 3)
        fea = round(num_fea / num, 3)
        rho = round(np.mean(rho), 3)
        tau = round(np.mean(tau), 3)
        pk10 = round(np.mean(pk10), 3)
        pk20 = round(np.mean(pk20), 3)

        rate = round(np.mean(rate), 3)
        recall = round(np.mean(recall), 3)
        precision = round(np.mean(precision), 3)
        f1 = round(np.mean(f1), 3)
        sim = round(np.mean(sim), 3)

        self.results.append(
            ('model_name', 'dataset', 'graph_set', '#testing_pairs', 'time_usage(s/100p)', 'mae','acc',
             'fea', 'rho', 'tau', 'pk10', 'pk20','precision', 'recall', 'f1'))
        self.results.append(
            (self.args.model_name, self.args.dataset, testing_graph_set, num, time_usage, mae_path, acc,fea,rho,tau,pk10,pk20,precision, recall, f1))

        print(*self.results[-2], sep='\t')
        print(*self.results[-1], sep='\t')
        with open(self.args.abs_path + self.args.result_path+self.args.dataset+'/results_'+self.args.model_name+'.txt', 'a') as f:
            print("## Post-processing", file=f)
            print("```", file=f)
            print(*self.results[-2], sep='\t', file=f)
            print(*self.results[-1], sep='\t', file=f)
            print("```\n", file=f)

    def score(self, testing_graph_set='test', test_k=None):
        print("\n\nModel evaluation on {} set.\n".format(testing_graph_set))
        if testing_graph_set == 'test':
            testing_graphs = self.testing_graphs
        elif testing_graph_set == 'test_small':
            testing_graphs = self.testing_graphs_small
        elif testing_graph_set == 'test_large':
            testing_graphs = self.testing_graphs_large
        elif testing_graph_set == 'test2':
            testing_graphs = self.testing2_graphs
        elif testing_graph_set == 'val':
            testing_graphs = self.val_graphs
        else:
            assert False

        self.model.eval()

        num = 0
        time_usage = []
        mse = []
        mae = []
        
        auc = []
        
        num_acc = 0
        num_fea = 0
        rho = []
        tau = []
        pk10 = []
        pk20 = []

        for pair_type, i, j_list in tqdm(testing_graphs, file=sys.stdout):
            pre = []
            gt = []
            t1 = time.time()
            for j in j_list:
                data = self.pack_graph_pair((pair_type, i, j))
                target, gt_ged = data["target"].item(), data["ged"]
                if test_k is None:
                    model_out = self.model(data)
                elif test_k == 0:
                    model_out = self.test_noah(data)
                elif test_k > 0:
                    model_out = self.test_matching(data, test_k)
                else:
                    assert False
                prediction, pre_ged = model_out[0], model_out[1]
                round_pre_ged = round(pre_ged)
                
                softmax_row = torch.nn.Softmax(dim=1)

                gt_mapping = data['mapping']
                pre_mapping = softmax_row(model_out[2])

                num += 1
                if prediction is None:
                    mse.append(-0.001)
                elif prediction.shape[0] == 1:
                    mse.append((prediction.item() - target) ** 2)
                else:
                    mse.append(F.mse_loss(prediction, data["ta_ged"]).item())
                pre.append(pre_ged)
                gt.append(gt_ged)
                
                mae.append(abs(round_pre_ged - gt_ged))
                if round_pre_ged == gt_ged:
                    num_acc += 1
                    num_fea += 1
                elif round_pre_ged > gt_ged:
                    num_fea += 1
            t2 = time.time()
            time_usage.append(t2 - t1)

            rho.append(spearmanr(pre, gt)[0])
            tau.append(kendalltau(pre, gt)[0])
            if rho[-1] != rho[-1]:
                rho[-1] = 0.
            if tau[-1] != tau[-1]:
                tau[-1] = 0.
            pk10.append(self.cal_pk(10, pre, gt))
            pk20.append(self.cal_pk(20, pre, gt))

        time_usage = round(np.mean(time_usage), 3)
        mse = round(np.mean(mse) * 1000, 3)
        mae = round(np.mean(mae), 3)
        acc = round(num_acc / num, 3)
        fea = round(num_fea / num, 3)

        rho = round(np.mean(rho), 3)
        tau = round(np.mean(tau), 3)
        pk10 = round(np.mean(pk10), 3)
        pk20 = round(np.mean(pk20), 3)

        self.results.append(
            ('model_name', 'dataset', 'graph_set', '#testing_pairs', 'time_usage(s/pair)', 'mse', 'mae', 'acc',
             'fea', 'rho', 'tau', 'pk10', 'pk20'))
        self.results.append((self.args.model_name, self.args.dataset, testing_graph_set, num, time_usage, mse, mae, acc,
                             fea, rho, tau, pk10, pk20))
        print(*self.results[-2], sep='\t')
        print(*self.results[-1], sep='\t')
        with open(self.args.abs_path + self.args.result_path+self.args.dataset+'/results_'+self.args.model_name+'.txt', 'a') as f:
            print("## Testing", file=f)
            print("```", file=f)
            print(*self.results[-2], sep='\t', file=f)
            print(*self.results[-1], sep='\t', file=f)
            print("```\n", file=f)

    def process(self, testing_graph_set='test'):
        print("\n\nModel evaluation on {} set.\n".format(testing_graph_set))
        if testing_graph_set == 'test':
            testing_graphs = self.testing_graphs
        elif testing_graph_set == 'test_small':
            testing_graphs = self.testing_graphs_small
        elif testing_graph_set == 'test_large':
            testing_graphs = self.testing_graphs_large
        elif testing_graph_set == 'test2':
            testing_graphs = self.testing2_graphs
        elif testing_graph_set == 'val':
            testing_graphs = self.val_graphs
        else:
            assert False

        num = 0
        time_usage = []
        mse = []
        mae = []
        num_acc = 0
        num_fea = 0
        rho = []
        tau = []
        pk10 = []
        pk20 = []

        for pair_type, i, j_list in tqdm(testing_graphs, file=sys.stdout):
            pre = []
            gt = []
            t1 = time.time()
            for j in j_list:
                data = self.pack_graph_pair((pair_type, i, j))
                target, gt_ged = data["target"].item(), data["ged"]
                gw = GEDGW(data, self.args)
                out1 =  gw.process()
                pre_ged = out1[1]
                prediction = None
                round_pre_ged = round(pre_ged)

                num += 1
                if prediction is None:
                    mse.append(-0.001)
                elif prediction.shape[0] == 1:
                    mse.append((prediction.item() - target) ** 2)
                else:
                    mse.append(F.mse_loss(prediction, data["ta_ged"]).item())
                pre.append(pre_ged)
                gt.append(gt_ged)

                mae.append(abs(round_pre_ged - gt_ged))
                if round_pre_ged == gt_ged:
                    num_acc += 1
                    num_fea += 1
                elif round_pre_ged > gt_ged:
                    num_fea += 1
            t2 = time.time()
            time_usage.append(t2 - t1)

            rho.append(spearmanr(pre, gt)[0])
            tau.append(kendalltau(pre, gt)[0])
            if rho[-1] != rho[-1]:
                rho[-1] = 0.
            if tau[-1] != tau[-1]:
                tau[-1] = 0.
            pk10.append(self.cal_pk(10, pre, gt))
            pk20.append(self.cal_pk(20, pre, gt))

        time_usage = round(np.mean(time_usage), 3)
        mse = round(np.mean(mse) * 1000, 3)
        mae = round(np.mean(mae), 3)
        acc = round(num_acc / num, 3)
        fea = round(num_fea / num, 3)

        rho = round(np.mean(rho), 3)
        tau = round(np.mean(tau), 3)
        pk10 = round(np.mean(pk10), 3)
        pk20 = round(np.mean(pk20), 3)

        self.results.append(
            ('model_name', 'dataset', 'graph_set', '#testing_pairs', 'time_usage(s/pair)', 'mse', 'mae', 'acc',
             'fea', 'rho', 'tau', 'pk10', 'pk20'))
        self.results.append((self.args.model_name, self.args.dataset, testing_graph_set, num, time_usage, mse, mae, acc,
                             fea, rho, tau, pk10, pk20))

        print(*self.results[-2], sep='\t')
        print(*self.results[-1], sep='\t')
        with open(self.args.abs_path + self.args.result_path+self.args.dataset+'/results_'+self.args.model_name+'.txt', 'a') as f:
            print("## Process", file=f)
            print("```", file=f)
            print(*self.results[-2], sep='\t', file=f)
            print(*self.results[-1], sep='\t', file=f)
            print("```\n", file=f)      

    def batch_score(self, testing_graph_set='test', test_k=100):
        print("\n\nModel evaluation on {} set.\n".format(testing_graph_set))
        if testing_graph_set == 'test':
            testing_graphs = self.testing_graphs
        elif testing_graph_set == 'test_small':
            testing_graphs = self.testing_graphs_small
        elif testing_graph_set == 'test_large':
            testing_graphs = self.testing_graphs_large
        elif testing_graph_set == 'test2':
            testing_graphs = self.testing2_graphs
        elif testing_graph_set == 'val':
            testing_graphs = self.val_graphs
        else:
            assert False

        self.model.eval()

        batch_results = []
        for pair_type, i, j_list in tqdm(testing_graphs, file=sys.stdout):
            res = []
            for j in j_list:
                data = self.pack_graph_pair((pair_type, i, j))
                gt_ged = data["ged"]
                time_list, pre_ged_list = self.test_matching(data, test_k, batch_mode=True)
                res.append((gt_ged, pre_ged_list, time_list))
            batch_results.append(res)

        batch_num = len(batch_results[0][0][1])
        for i in range(batch_num):
            time_usage = []
            num = 0 
            mse = []
            mae = []
            num_acc = 0
            num_fea = 0
            num_better = 0
            ged_better = 0.
            rho = []
            tau = []
            pk10 = []
            pk20 = []

            for res_id, res in enumerate(batch_results):
                pre = []
                gt = []
                for gt_ged, pre_ged_list, time_list in res:
                    time_usage.append(time_list[i])
                    pre_ged = pre_ged_list[i]
                    round_pre_ged = round(pre_ged)

                    num += 1
                    mse.append(-0.001)
                    pre.append(pre_ged)
                    gt.append(gt_ged)

                    mae.append(abs(round_pre_ged - gt_ged))
                    if round_pre_ged == gt_ged:
                        num_acc += 1
                        num_fea += 1
                    elif round_pre_ged > gt_ged:
                        num_fea += 1
                    else:
                        num_better += 1
                        ged_better += (gt_ged - round_pre_ged)
                rho.append(spearmanr(pre, gt)[0])
                tau.append(kendalltau(pre, gt)[0])
                pk10.append(self.cal_pk(10, pre, gt))
                pk20.append(self.cal_pk(20, pre, gt))

            time_usage = round(np.mean(time_usage), 3)
            mse = round(np.mean(mse) * 1000, 3)
            mae = round(np.mean(mae), 3)
            acc = round(num_acc / num, 3)
            fea = round(num_fea / num, 3)
            rho = round(np.mean(rho), 3)
            tau = round(np.mean(tau), 3)
            pk10 = round(np.mean(pk10), 3)
            pk20 = round(np.mean(pk20), 3)
            if num_better > 0:
                avg_ged_better = round(ged_better / num_better, 3)
            else:
                avg_ged_better = None
            self.results.append((self.args.model_name, self.args.dataset, testing_graph_set, num, time_usage, mse, mae, acc,
                                 fea, rho, tau, pk10, pk20, num_better, avg_ged_better))

            print(*self.results[-1], sep='\t')
            with open(self.args.abs_path + self.args.result_path+self.args.dataset+'/results_'+self.args.model_name+'.txt', 'a') as f:
                print(*self.results[-1], sep='\t', file=f)

    def print_results(self):
        for r in self.results:
            print(*r, sep='\t')

        with open(self.args.abs_path + self.args.result_path+self.args.dataset+'/results_'+self.args.model_name+'.txt', 'a') as f:
            for r in self.results:
                print(*r, sep='\t', file=f)

    @staticmethod
    def data_to_nx(edges, features):
        edges = edges.t().tolist()

        nx_g = nx.Graph()
        n, num_label = features.shape

        if num_label == 1:
            labels = [-1 for i in range(n)]
        else:
            labels = [-1] * n
            for i in range(n):
                for j in range(num_label):
                    if features[i][j] > 0.5:
                        labels[i] = j
                        break

        for i, label in enumerate(labels):
            nx_g.add_node(i, label=label)

        for u, v in edges:
            if u < v:
                nx_g.add_edge(u, v)
        return nx_g

    def test_matching(self, data, test_k,test_k_GW=100):
        g1 = dgl.graph((data["edge_index_1"][0], data["edge_index_1"][1]), num_nodes=data["n1"])
        g2 = dgl.graph((data["edge_index_2"][0], data["edge_index_2"][1]), num_nodes=data["n2"])
        g1.ndata['f'] = data["features_1"]
        g2.ndata['f'] = data["features_2"]
        _, pre_ged, soft_matrix = self.model(data)
        if self.args.model_name=="Grasp":
            m = torch.nn.Softmax(dim=1)
            soft_matrix = (m(soft_matrix) * 1e9 + 1).round()
        solver = KBestMSolver(soft_matrix, g1, g2)
        solver.get_matching(test_k)
        min_res = solver.min_ged
        if self.args.GW:
            if min_res>min_res2:
                return None, min_res2, best_matching2
        best_matching = solver.best_matching()
        return None, min_res, best_matching

    def prediction_analysis(self, values, info_str=''):
        if not self.args.prediction_analysis:
            return
        neg_num = 0
        pos_num = 0
        pos_error = 0.
        neg_error = 0.
        for v in values:
            if v >= 0:
                pos_num += 1
                pos_error += v
            else:
                neg_num += 1
                neg_error += v

        tot_num = neg_num + pos_num
        tot_error = pos_error - neg_error

        pos_error = round(pos_error / pos_num, 3) if pos_num > 0 else None
        neg_error = round(neg_error / neg_num, 3) if neg_num > 0 else None
        tot_error = round(tot_error / tot_num, 3) if tot_num > 0 else None

        with open(self.args.abs_path + self.args.result_path+self.args.dataset+'/results_'+self.args.model_name+'.txt', 'a') as f:
            print("prediction_analysis", info_str, sep='\t', file=f)
            print("num", pos_num, neg_num, tot_num, sep='\t', file=f)
            print("err", pos_error, neg_error, tot_error, sep='\t', file=f)
            print("--------------------", file=f)

    def save(self, epoch):
        torch.save(self.model.state_dict(),
                   self.args.abs_path + self.args.model_path+self.args.model_name+'/' + self.args.dataset + '_' + str(epoch))

    def load(self, epoch):
        model_dicts={"Grasp":"Grasp"}
        self.model.load_state_dict(
            torch.load(self.args.abs_path + self.args.model_path +model_dicts[self.args.model_name]+'/'+ self.args.dataset + '_' + str(epoch)))

