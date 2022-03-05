import node2vec
import numpy as np
import networkx as nx
from gensim.models import Word2Vec

is_directed = True
p = 2
q = 1
num_walks = 100
walk_length = 80
dimensions = 64
window_size = 10
iter = 100
Adj_file = "/home/wangao/Traffic_prediction_with_missing_value/Baseline_model/Prediction_model/" + "GMAN_nav-beijing/data/Adj(nav-beijing).txt"
SE_file = "/home/wangao/Traffic_prediction_with_missing_value/Baseline_model/Prediction_model/" + "GMAN_nav-beijing/data/SE(nav-beijing).txt"
# Adj_file = "/home/wangao/Traffic_prediction_with_missing_value/Baseline_model/Prediction_model/" + "GMAN_METR/data/Adj(METR).txt"
# SE_file = "/home/wangao/Traffic_prediction_with_missing_value/Baseline_model/Prediction_model/" + "GMAN_PEMS(M)/data/SE(PEMS(M)).txt"
csv_file = "/data/wangao/" + "nav-beijing/dataset/W_1362.npy"

def read_graph(edgelist):
    G = nx.read_edgelist(
        edgelist, nodetype=int, data=(('weight',float),),
        create_using=nx.DiGraph())

    return G

def learn_embeddings(walks, dimensions, output_file):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        sentences=walks, vector_size = dimensions, window = 10, min_count=0, sg=1,
        workers = 8, epochs = iter)
    model.wv.save_word2vec_format(output_file)
	
    return
def write_graph(adj, Adj_file):
    with open(Adj_file, "w") as f:
        for i in range(adj.shape[0]):
            for j in range(adj.shape[0]):
                f.writelines("%d %d %.4f\n" % (i,j,adj[i][j]))

# adj = np.load(csv_file)
# write_graph(adj, Adj_file)
# print("writs finish")
nx_G = read_graph(Adj_file)
G = node2vec.Graph(nx_G, is_directed, p, q)
G.preprocess_transition_probs()
walks = G.simulate_walks(num_walks, walk_length)
learn_embeddings(walks, dimensions, SE_file)
