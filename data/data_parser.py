import os
import sys
import copy
import random
import numpy as np
import scipy.sparse as sp
from goatools.obo_parser import GODag

import matplotlib.pyplot as plt


class Parser():

    def __init__(self):
        self.infile = None
        self.outdir = None
    
    def build(self):
        raise NotImplementedError

    def write(self):
        raise NotImplementedError

    def set_infile(self, infile):
        if os.path.exists(infile):
            self.infile = os.path.abspath(infile)
        else:
            raise FileNotFoundError
    
    def set_outdir(self, outdir):
        if os.path.exists(outdir):
            self.outdir = os.path.abspath(outdir)
        else:
            os.mkdir(outdir)

    def get_infile(self):
        return self.infile

    def get_outdir(self):
        return self.outdir


class GOParser(Parser):

    def __init__(self, 
                 basic_rels=("is_a", "part_of", "regulates", "positively_regulates", "negatively_regulates")):
        super(GOParser, self).__init__()
        self.emap = None
        self.rmap = None
        self.relationships = {}
        self.golist = []

        self.infile = None
        self.outdir = None

        self.include_rels = basic_rels

    def build(self):
        self.ent_file = self.outdir + "/entity2id.txt"
        self.rel_file = self.outdir + "/relation2id.txt"
        self.train_file = self.outdir + "/train2id.txt"
        # self.valid_file = self.outdir + "/valid2id.txt"
        # self.test_file = self.outdir + "/test2id.txt"

        self._init_golist()
        self._extract_relationship()
        # self.relationships = self.include_rels
        self._init_emap()
        self._init_rmap()
        self.train2id = self.construct_KGDatasets()

    def _init_golist(self):
        godag = GODag(obo_file=self.infile, optional_attrs="relationship")
        golist = list(set(godag.values()))
        # golist = [val for val in godag.values if val.namespace == "biological_process"]
        self.golist = golist

    def _extract_relationship(self):
        self.relationships["is_a"] = 0
        for term in self.golist:
            for _ in term.parents:
                self.relationships["is_a"] += 1
            if term.relationship:
                for rel in term.relationship.keys():
                    if rel not in self.relationships.keys():
                        self.relationships[rel] = 0
                    self.relationships[rel] += 1

    def _init_emap(self):
        if self.emap is not None:
            return
        self.emap = dict()
        for id, ent in enumerate(self.golist):
            self.emap[ent.id] = str(id)

    def _init_rmap(self):
        self.rmap = dict()
        for id, rel in enumerate(self.relationships.keys()):
            self.rmap[rel] = str(id)

    def write_entity2id(self):
        with open(self.ent_file, 'w') as f:
            f.write(str(len(self.emap))+"\n")
            for entity, eid in self.emap.items():
                f.write(entity+"\t"+eid+"\n")

    def write_relation2id(self):
        with open(self.rel_file, "w") as f:
            f.write(str(len(self.rmap))+"\n")
            for rel, rid in self.rmap.items():
                f.write(rel+"\t"+rid+"\n")

    def write(self):
        self.write_entity2id()
        self.write_relation2id()

        with open(self.train_file, 'w') as f:
            f.write(str(len(self.train2id))+"\n")
            for triple in self.train2id:
                h, r, t = triple
                f.write(h+"\t"+r+"\t"+t+"\n")

        # with open(self.valid_file, "w") as f:
        #     f.write(str(len(self.valid2id))+"\n")
        #     for triple in self.valid2id:
        #         h, r, t = triple
        #         f.write(h+"\t"+r+"\t"+t+"\n")

        # with open(self.test_file, "w") as f:
        #     f.write(str(len(self.test2id))+"\n")
        #     for triple in self.test2id:
        #         h, r, t = triple
        #         f.write(h+"\t"+r+"\t"+t+"\n")

        print("GOParser: Done!")
    
    def construct_KGDatasets(self, split=False):
        # Triple = (h, r, t)
        # cross link: (BP2MF, 836), (MF2BP, 1052)
        triple_list = list()
        for cur in self.golist:
            if cur.namespace != "biological_process":
                continue
            triple_list.extend((self.emap[cur.id], self.rmap['is_a'], self.emap[parent.id]) for parent in cur.parents)
            if cur.relationship:
                for relation in cur.relationship.keys():
                    if relation not in self.relationships.keys():
                        continue
                    else:
                        for tail in cur.relationship[relation]:
                            if tail.namespace != "biological_process":
                                continue
                            triple_list.append((self.emap[cur.id], self.rmap[relation], self.emap[tail.id]))

        if split:
            train2id, valid2id, test2id = self._split(triple_list)
            return train2id, valid2id, test2id
        else:
            return triple_list

    def _split(self, triple_list):
        # split datasets into Train(80%) | Valid(10%) | Test(10%)
        tri_size = len(triple_list)
        random.shuffle(triple_list)
        train2id = triple_list[:int((tri_size+1)*.80)]
        valid2id = triple_list[int((tri_size+1)*.80):int((tri_size+1)*.90)]
        test2id = triple_list[int((tri_size+1)*.90):]
        train2id, valid2id, test2id = self._move_unseen_to_train(train2id, valid2id, test2id)

        return train2id, valid2id, test2id
        
    def _move_unseen_to_train(self, train, valid, test):
        # Move unseen entities in valid / test sets to train sets
        # !WARNING: NOT operate list in for loop
        train_ent = set()
        train_ = copy.deepcopy(train)
        valid_ = list()
        test_ = list()

        for triplet in train:
            h, _, t = triplet
            train_ent.add(h)
            train_ent.add(t)

        for triplet in valid:
            h, _, t = triplet
            if h in train_ent and t in train_ent:
                valid_.append(triplet)
            else:
                train_.append(triplet)
        
        for triplet in test:
            h, _, t = triplet
            if h in train_ent and t in train_ent:
                test_.append(triplet)
            else:
                train_.append(triplet)
                
        return train_, valid_, test_

    def set_emap(self, emap):
        self.emap = emap

    def set_rmap(self, rmap):
        self.rmap = rmap


class HNParser(Parser):

    def __init__(self, weighted=True):
        super(HNParser, self).__init__()
        self.gmap = None
        self.adj = None

        self.infile = None
        self.outdir = None

        self.weighted = weighted
    
    def build(self):
        self.gmap_file = self.outdir + "/genexc2id.txt"
        self.hn_file = self.outdir + "/hnetxc.npz"

        edges = self._init_edges()
        self._init_gmap(edges)
        self._init_adj(edges)

    def write(self):
        self.write_gene2id()
        sp.save_npz(self.hn_file, self.adj)
        print("HNParser: Done!")

    def _init_edges(self):
        edges = []
        with open(self.infile, "r") as f:
            f.readline()
            for line in f:
                i, j, w = line.strip().split()
                if self.weighted:
                    edges.append((i, j, w))
                else:
                    edges.append((i, j, 1))
        return edges

    def _init_gmap(self, edges):
        if self.gmap != None:
            return
        else:
            self.gmap = dict()
            genes = set()
            for tri in edges:
                i, j, _ = tri
                genes.add(i)
                genes.add(j)
            for id, gene in enumerate(genes):
                self.gmap[gene] = str(id)

    def _init_adj(self, edges):
        # Construct undirected graph
        if self.adj != None:
            return
        else:
            row = []
            col = []
            data = []
            for tri in edges:
                i, j, w = tri
                if i not in self.gmap.keys():
                    continue
                if j not in self.gmap.keys():
                    continue
                
                i = (int)(self.gmap[i])
                j = (int)(self.gmap[j])
                w = (float)(w)
                row.append(i)
                col.append(j)
                data.append(w)

            self.adj = sp.coo_matrix((data, (row, col)), shape=(len(self.gmap), len(self.gmap)))

    def write_gene2id(self):
        with open(self.gmap_file, 'w') as f:
            f.write(str(len(self.gmap))+"\n")
            for gene, gid in self.gmap.items():
                f.write(gene+"\t"+str(gid)+"\n")

    def set_gmap(self, gmap):
        self.gmap = gmap

    def get_gmap(self):
        return self.gmap

    def get_adj(self):
        return self.adj


class DGNParser(Parser):

    def __init__(self):
        super(DGNParser, self).__init__()
        self.dmap = None
        self.gmap = None
        self.d2g = None

    def build(self):
        self.dmap_file = self.outdir + "/dis2id_xc.txt"
        self.d2g_file = self.outdir + "/d2g_xc.npz"

        self._init_dmap()
        self._init_d2g()

    def write(self):
        self.write_dmap()
        sp.save_npz(self.d2g_file, self.d2g)
        print("DGNParser: Done!")

    def _init_dmap(self):
        if self.dmap != None:
            return
        else:
            self.dmap = dict()
            dis_list = set()
            with open(self.infile, "r") as f:
                f.readline()
                for line in f:
                    line = line.strip().split("\t")
                    dis = line[4]
                    dis_list.add(dis)
            for id, dis in enumerate(dis_list):
                self.dmap[dis] = str(id)

    def _init_d2g(self):
        row = []
        col = []
        with open(self.infile, "r") as f:
            f.readline()
            for line in f:
                line = line.strip().split("\t")
                gene, dis = line[0], line[4]
                if gene not in self.gmap.keys():
                    # filter out gene not include in the common set 
                    continue
                else: 
                    # generate disease2gene pair
                    dis = self.dmap[dis]
                    gene = self.gmap[gene]
                    row.append(dis)
                    col.append(gene)
            
        self.d2g = sp.coo_matrix((np.ones((len(row))), (row, col)), shape=(len(self.dmap), len(self.gmap)))

    def write_dmap(self):
        with open(self.dmap_file, 'w') as f:
            f.write(str(len(self.dmap))+"\n")
            for dis, did in self.dmap.items():
                f.write(dis+"\t"+str(did)+"\n")

    def set_gmap(self, gmap):
        self.gmap = gmap

    def set_dmap(self, dmap):
        self.dmap = dmap

    def get_gmap(self):
        return self.gmap

    def get_dmap(self):
        return self.dmap


def read_gene2go(file, emap, gmap, tax_id=9606):
    # align gene and ontology
    row = []
    col = []

    with open(file, "r") as f:
        f.readline()
        for line in f:
            tax, gene, go,  *_ = line.strip().split("\t")
            if (int)(tax) == tax_id:
                if gene in gmap.keys() and go in emap.keys():
                    row.append((int)(gmap[gene]))
                    col.append((int)(emap[go]))
                else:
                    continue

    g2o = sp.coo_matrix((np.ones((len(row))), (row, col)), shape=(len(gmap), len(emap)))
    return g2o

def write_gene2go(file, g2o):
    sp.save_npz(file, g2o)
    print("Write G2O file!")

def read_mapping(file):
    xmap = dict()
    with open(file, "r") as f:
        f.readline()
        for line in f:
            x, idx = line.strip().split()
            xmap[x] = idx
    return xmap

def main():
    # use HumanNet as background graph
    data_dir = os.path.dirname(__file__)
    # emap = read_mapping(data_dir+"/entity2id.txt")
    gmap = read_mapping(data_dir+"/gene2id.txt")
    # dmap = read_mapping(data_dir+"/dis2id.txt")

    gop = GOParser()
    gop.set_infile(data_dir+"/raw/go.obo")
    gop.set_outdir(data_dir+"/BP")
    # gop.set_emap(emap)
    gop.build()
    emap = gop.emap
    gop.write()
    
    # hnp = HNParser()
    # hnp.set_infile(data_dir+"/raw/HumanNet-XC.tsv")
    # hnp.set_outdir(data_dir+"/XC")
    # hnp.set_gmap(gmap)
    # hnp.build()
    # gmap = hnp.gmap
    # hnp.write()

    # dgp = DGNParser()
    # dgp.set_infile(data_dir+"/raw/all_gene_disease_associations.tsv")
    # dgp.set_outdir(data_dir+"/")
    # dgp.set_gmap(gmap)
    # dgp.set_dmap(dmap)
    # dgp.build()
    # dgp.write()


    g2o = read_gene2go(data_dir+"/raw/gene2go", emap, gmap)
    write_gene2go(data_dir+"/BP/g2o.npz", g2o)
    print("Finished!")

    # ddegree = np.zeros(len(dmap))
    # gdegree = np.zeros(len(gmap))
    # for (d, g) in dgpair:
    #     ddegree[(int)(d)] += 1
    #     gdegree[(int)(g)] += 1
    # uniqued, countd = np.unique(ddegree, return_counts=True)
    # uniqueg, countg = np.unique(gdegree, return_counts=True)

    # plt.figure(figsize=(12, 8))
    # plt.loglog(uniqueg, countg, 'b.')
    # plt.xlabel("Num of gene related diseases")
    # plt.ylabel("Counts")
    # plt.title("Gene related Diseases distribution")
    # plt.show()

def construct_gene2id():
    data_dir = os.path.dirname(__file__)
    g2o_list = set()
    hn_list = set()
    dgn_list = set()
    with open(data_dir+"/raw/gene2go", "r") as f:
        f.readline()
        for line in f:
            tax, gene, go,  *_ = line.strip().split("\t")
            if (int)(tax) == 9606:
                g2o_list.add(gene)
    with open(data_dir+"/raw/HumanNet-XN.tsv") as f:
        f.readline()
        for line in f:
            i, j, _ = line.strip().split("\t")
            hn_list.add(i)
            hn_list.add(j)
    with open(data_dir+"/raw/all_gene_disease_associations.tsv") as f:
        f.readline()
        for line in f:
            line = line.strip().split("\t")
            gene = line[0]
            dgn_list.add(gene)
    common = []
    for g in g2o_list:
        if g in hn_list and g in dgn_list:
            common.append(g)
    print(len(common))
    with open(data_dir+"/gene2id.txt", "w") as f:
        f.write(str(len(common))+"\n")
        for id, gene in enumerate(common):
            f.write(gene+"\t"+str(id)+"\n")
        
if __name__ == "__main__":
    # construct_gene2id()
    main()