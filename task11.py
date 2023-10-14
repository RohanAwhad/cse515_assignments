#!python3

import config
import helper
from feature_descriptor import FeatureDescriptor
import numpy as np
import networkx as nx
import pickle
from tqdm import tqdm

from similarity_metrics import get_similarity, get_similarity_mat_x_mat, get_top_k_ids_n_scores


def main():
    feature_descriptor = FeatureDescriptor(config.RESNET_MODEL)
    inp = helper.get_user_input('feat_space,n,m,label',len(config.DATASET), len(set(config.DATASET.y)))

    # Create Similarity Graph, for each image calculate n most similar images.
    _tmp = config.FEAT_DESC_FUNCS[inp['feat_space']]
    feat_db = _tmp[config.FEAT_DB]
    feat_idx = _tmp[config.IDX]
    similarity_metric = _tmp[config.SIMILARITY_METRIC]

    G = nx.Graph()
    p = np.repeat(0, len(feat_idx))
    # For each image id, create a new node and create edges with its top n images. 
    for id in tqdm(feat_idx):
        # Personalized Page Rank, focus on the label. If label is matched give 1 in the teleportation vector. Otherwise leave it to zero
        if inp['label'] == feat_idx[id][1]:
            p[id] = 1
        # Add new vertex for the image
        G.add_node(feat_idx[id][0])
        img = config.DATASET[feat_idx[id][0]][0]
        if img.mode != 'RGB': img = img.convert('RGB')
        # Extract image features
        query_feat = feature_descriptor.extract_features(img, inp['feat_space'])
        # Get similarity scores of the current image with all other images.
        similarity_scores = get_similarity(query_feat, feat_db, similarity_metric)
        # Find top n images.
        top_n_ids, top_n_scores = get_top_k_ids_n_scores(similarity_scores, feat_idx, inp['n'])
        for top_id in top_n_ids:
            # Add an edge with top n images.
            G.add_edge(feat_idx[id][0], top_id)
    # Considering alpha to be 0.85.
    alpha = 0.85
    # Persist the graph so that it can be loaded back if needed.
    with open(f'{inp["feat_space"]}_{inp["n"]}_{inp["m"]}.gpickle', 'wb') as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
    # Calculate page rank measure.
    def google_matrix(G, p, alpha=0.85):
        # Adjacency matrix
        M = np.asmatrix(nx.to_numpy_array(G))
        N = len(G)
        if N == 0:
            return M
        # Transition matrix
        M /= M.sum(axis=1) 
        return alpha * M + (1 - alpha) * p
    # Normalize teliportation vector based on seed set
    p = p / np.count_nonzero(p == 1)
    M = google_matrix(G, p, alpha)
    # Find eigen values and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(M.T)
    ind = np.argmax(eigenvalues)
    largest = np.array(eigenvectors[:, ind]).flatten().real
    norm = float(largest.sum())
    largest = largest / norm
    ranks_ev = dict(zip(G, map(float, largest / norm)))
    top_m_ids = sorted(ranks_ev, key=ranks_ev.get, reverse=True)[:inp['m']]
    top_m_imgs = [config.DATASET[x][0] for x in top_m_ids]
    for i,image in enumerate(top_m_imgs):
        image.save('./Outputs/'+str(top_m_ids[i])+'.jpg')

if __name__ == '__main__':
  while True: main()