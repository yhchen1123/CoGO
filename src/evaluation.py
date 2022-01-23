import os

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn.functional as F

from sklearn import metrics
from xlrd import open_workbook


root = os.path.dirname(os.path.dirname(__file__))
path_dis2id = root + "/data/dis2id.txt"
path_ent2id = root + "/data/entity2id.txt"
path_pos = root + "/data/positive.xlsx"
path_neg1 = root + "/data/random1.xlsx"
path_neg2 = root + "/data/random2.xlsx"

########################################################################
# Disease Similarity Evaluation
########################################################################


def load_d2g(path):
    return sp.load_npz(path).todense()

# load dis2id
def load_dmap(path):
    dmap = dict()
    inv = dict()
    with open(path) as f:
        f.readline()
        for line in f:
            dis, id = line.strip().split()
            dmap[dis] = (int)(id)
            inv[(int)(id)] = dis
    return dmap, inv

def load_disid(path):
    idmap = dict()
    with open(path, "r") as f:
        f.readline()
        for line in f:
            line = line.strip().split("\t")
            disid, name = line[0], line[1]
            if disid not in idmap.keys():
                idmap[disid] = name
            else:
                continue
    return idmap

# load /data/.xlsx
def load_xlsx(path, dmap):
    data = open_workbook(path)
    table = data.sheets()[0]
    dataset = []
    for i in range(table.nrows):
        dis1 = table.cell(i, 1).value
        dis2 = table.cell(i, 3).value
        try:
            d1 = (int)(dmap[dis1])
            d2 = (int)(dmap[dis2])
            dataset.append((d1, d2))
        except:
            continue

    return dataset

def get_disease_auc(dis_embeddings, dis_path, plot=False):
    """
    Calculate disease similarity AUC metrics
    :param dis_sim: disease embeddings matrix
    :param plot: whether to draw ROC
    :return: AUROC, AP score
    """
    dmap, _ = load_dmap(dis_path+"/dis2id.txt")
    pos_data = load_xlsx(path_pos, dmap)
    neg_data1 = load_xlsx(path_neg1, dmap)
    neg_data2 = load_xlsx(path_neg2, dmap)
    data = pos_data + neg_data1 + neg_data2
    data = np.asarray(data)

    y = np.zeros(len(data), dtype=np.int32)
    for i in range(len(pos_data)):
        y[i] = 1

    x = F.cosine_similarity(dis_embeddings[data[:, 0]], dis_embeddings[data[:, 1]]).view(-1).numpy()

    auroc = metrics.roc_auc_score(y, x)
    ap = metrics.average_precision_score(y, x)

    return auroc, ap

def topk(dis_embeddings, dis_path, k):
    """ Find topk similar diseases for each anchor. """
    dmap, inv = load_dmap(dis_path+"/dis2id.txt")
    idmap = load_disid(root+"/data/raw/disease_mappings_to_attributes.tsv")
    pos_data = load_xlsx(path_pos, dmap)
    anchor = set(list(zip(*pos_data))[0])
    anchor_index = torch.tensor([i for i in anchor])
    
    emb_norm = F.normalize(dis_embeddings, dim=1)
    res = torch.mm(emb_norm, emb_norm.T).fill_diagonal_(0.)
    _, topk_index = res[anchor_index].topk(k)
    anchor = [idmap[inv[i]] for i in anchor_index.tolist()]
    top = [[idmap[inv[k]] for k in j] for j in topk_index.tolist()]
    return anchor, top