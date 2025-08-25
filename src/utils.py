from os.path import basename, isfile
from os import makedirs
from glob import glob
import networkx as nx
import json
from texttable import Texttable

def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    rows = [["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys]
    t.add_rows(rows)
    print(t.draw())

def sorted_nicely(l):
    def tryint(s):
        try:
            return int(s)
        except:
            return s
    import re
    def alphanum_key(s):
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    return sorted(l, key=alphanum_key)

def get_file_paths(dir, file_format='json'):
    dir = dir.rstrip('/')
    paths = sorted_nicely(glob(dir + '/*.' + file_format))
    return paths

def iterate_get_graphs(dir, file_format):
    assert file_format in ['gexf', 'json', 'onehot', 'anchor']
    graphs = []
    for file in get_file_paths(dir, file_format):
        gid = int(basename(file).split('.')[0])
        if file_format == 'gexf':
            g = nx.read_gexf(file)
            g.graph['gid'] = gid
            if not nx.is_connected(g):
                raise RuntimeError('{} not connected'.format(gid))
        elif file_format == 'json':
            g = json.load(open(file, 'r'))
            g['gid'] = gid
        elif file_format in ['onehot', 'anchor']:
            g = json.load(open(file, 'r'))
        graphs.append(g)
    return graphs

def load_all_graphs(data_location, dataset_name):
    graphs = iterate_get_graphs(data_location + "json_data/" + dataset_name + "/train", "json")
    train_num = len(graphs)
    graphs += iterate_get_graphs(data_location + "json_data/" + dataset_name + "/test", "json")
    test_num = len(graphs) - train_num
    val_num = test_num
    train_num -= val_num
    return train_num, val_num, test_num, graphs

def load_labels(data_location, dataset_name):
    path = data_location + "json_data/" + dataset_name + "/labels.json"
    global_labels = json.load(open(path, 'r'))
    features = iterate_get_graphs(data_location + "json_data/" + dataset_name + "/train", "onehot") \
             + iterate_get_graphs(data_location + "json_data/" + dataset_name + "/test", "onehot")
    print('Load one-hot label features (dim = {}) of {}.'.format(len(global_labels), dataset_name))
    return global_labels, features

def load_ged(ged_dict, data_location='', dataset_name='AIDS', file_name='TaGED.json'):
    path = "{}json_data/{}/{}".format(data_location, dataset_name, file_name)
    TaGED = json.load(open(path, 'r'))
    for (id_1, id_2, ged_value, ged_nc, ged_in, ged_ie, mappings) in TaGED:
        ta_ged = (ged_value, ged_nc, ged_in, ged_ie)
        ged_dict[(id_1, id_2)] = (ta_ged, mappings)

def load_features(data_location, dataset_name, feature_name):
    features = iterate_get_graphs(data_location + "json_data/" + dataset_name + "/train", feature_name) \
             + iterate_get_graphs(data_location + "json_data/" + dataset_name + "/test", feature_name)
    feature_dim = len(features[0][0])
    print('Load {} features (dim = {}) of {}.'.format(feature_name, feature_dim, dataset_name))
    return feature_dim, features