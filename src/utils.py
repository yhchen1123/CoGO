import torch
import numpy as np
import scipy.sparse as sp


########################################################################
# Sparse Matrix Utils
########################################################################

def load_sparse(path):
    return sp.load_npz(path).tocoo()

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mx_to_torch_sparse_tesnsor(mx):
    """Convert scipy sparse coo matrix to a torch sparse tensor"""
    sparse_mx = mx.astype(np.float32)
    sparse_mx.eliminate_zeros()
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    size = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, size)

def generate_sparse_one_hot(num_ents, dtype=torch.float32):
    """ Creates a two-dimensional sparse tensor with ones along the diagnoal as one-hot encoding. """
    diag_size = num_ents
    diag_range = list(range(num_ents))
    diag_range = torch.tensor(diag_range)

    return torch.sparse_coo_tensor(
        indices=torch.vstack([diag_range, diag_range]),
        values=torch.ones(diag_size, dtype=dtype),
        size=(diag_size, diag_size))


########################################################################
# Knowledge Graph Utils
########################################################################

def load_triples(path):
    """Load knowledge graphs for RGAE model
    
    :param path: dir path of data file
    :return: train/valid/test datasets
    """
    train_total = 0
    triples = []
    train_file = path + "/train2id.txt"

    def load(file):
        triples = []
        with open(file, "r") as f:
            total = (int)(f.readline())
            for line in f:
                line = line.strip().split()
                h, r, t = line
                triples.append(((int)(h), (int)(r), (int)(t)))
        return total, triples

    train_total, triples = load(train_file)

    print("GO(%d) datasets loaded." % (train_total))

    return triples

def generate_inverses(triples, num_rels):
    """ Generate inverse relations """

    # Swap around head and tail. Create new relation ids for inverse relations.
    inverse_relations = torch.cat([triples[:, 2, None], triples[:, 1, None] + num_rels, triples[:, 0, None]], dim=1)
    assert inverse_relations.size() == triples.size()

    return inverse_relations

def generate_self_loops(num_ents, num_rels, device='cpu'):
    """ Generates self-loop triples and then applies edge dropout """

    # Create a new relation id for self loop relation.
    all = torch.arange(num_ents, device=device)[:, None]
    id  = torch.empty(size=(num_ents, 1), device=device, dtype=torch.long).fill_(2*num_rels)
    self_loops = torch.cat([all, id, all], dim=1)
    assert self_loops.size() == (num_ents, 3)

    return self_loops

def add_inverse_and_self(triples, num_ents, num_rels, device='cpu'):
    """ Adds inverse relations and self loops to a tensor of triples """

    # Swap around head and tail. Create new relation ids for inverse relations.
    inverse_relations = torch.cat([triples[:, 2, None], triples[:, 1, None] + num_rels, triples[:, 0, None]], dim=1)
    assert inverse_relations.size() == triples.size()

    # Create a new relation id for self loop relation.
    all = torch.arange(num_ents, device=device)[:, None]
    id  = torch.empty(size=(num_ents, 1), device=device, dtype=torch.long).fill_(2*num_rels)
    self_loops = torch.cat([all, id, all], dim=1)
    assert self_loops.size() == (num_ents, 3)

    return torch.cat([triples, inverse_relations, self_loops], dim=0)

def get_kg_data(triples, num_rels):
    triples = torch.tensor(triples, dtype=torch.long)
    inverse_triples = generate_inverses(triples, num_rels)
    triples = torch.cat([triples, inverse_triples], dim=0)
    edge_index = torch.cat([triples[:, 0, None], triples[:, 2, None]], dim=1).permute(1, 0)
    edge_type = triples[:, 1, None].view(-1)
    return edge_index, edge_type