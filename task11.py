#!python3

import config
import helper
from feature_descriptor import FeatureDescriptor
import numpy as np
import networkx as nx
import pickle
from tqdm import tqdm
import torch

from similarity_metrics import get_similarity, get_similarity_mat_x_mat, get_top_k_ids_n_scores
from task5 import get_label_vecs


def main():
    feature_descriptor = FeatureDescriptor(config.RESNET_MODEL)
    inp = helper.get_user_input('LORF,feat_space,n,m,label',len(config.DATASET), len(set(config.DATASET.y)))
    _tmp = config.FEAT_DESC_FUNCS[inp['feat_space']]
    
    feat_idx = _tmp[config.IDX]
    similarity_metric = _tmp[config.SIMILARITY_METRIC]

    if inp['LORF'] == 1:
        feat_temp = _tmp[config.FEAT_DB]
        inp_LS = helper.get_user_input('task_id,K_latent',len(config.DATASET), len(set(config.DATASET.y)))
        if(inp_LS['task_id']==3):
            inp_2 = helper.get_user_input('dim_red',len(config.DATASET))
            inp['dim_red'] = inp_2['dim_red']
            latent_space = helper.load_semantics("task" + str(inp_LS['task_id'])+ "_" + inp['feat_space']+"_"+inp['dim_red']+"_"+str(inp_LS['K_latent'])+".pkl")

        elif(inp_LS['task_id']==4):
            inp['dim_red']="cp"
            latent_space = helper.load_semantics("task" + str(inp_LS['task_id'])+ "_" + inp['feat_space']+"_"+inp['dim_red']+"_"+str(inp_LS['K_latent'])+"_feature.pkl")            

        elif(inp_LS['task_id']==5):
            inp_2 = helper.get_user_input('dim_red',len(config.DATASET))
            inp['dim_red'] = inp_2['dim_red']
            label_feat_sp = get_label_vecs(inp['feat_space']) #num_labelsX#num_features
            latent_space = helper.load_semantics("task" + str(inp_LS['task_id']) + "_label_label_simi_mat"+ "_" + inp['feat_space']+"_"+inp['dim_red']+"_"+str(inp_LS['K_latent'])+".pkl")
            feat_temp = get_similarity_mat_x_mat(feat_temp, label_feat_sp, similarity_metric)
            
        elif(inp_LS['task_id']==6):
            inp_2 = helper.get_user_input('dim_red',len(config.DATASET))
            inp['dim_red'] = inp_2['dim_red']
            latent_space = helper.load_semantics("task" + str(inp_LS['task_id']) + "_img_img_simi_mat"+ "_" + inp['feat_space']+"_"+inp['dim_red']+"_"+str(inp_LS['K_latent'])+".pkl")
            feat_temp = get_similarity_mat_x_mat(feat_temp, feat_temp, similarity_metric)
        
        feat_db = np.dot(feat_temp, np.transpose(latent_space))
        feat_db = torch.tensor(feat_db)
    else:
        feat_db = _tmp[config.FEAT_DB]
    
    # Create Similarity Graph, for each image calculate n most similar images.
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
        if inp['LORF'] == 1:
            if inp_LS['task_id']==5 or inp_LS['task_id']==6:
                query_feat = get_similarity_mat_x_mat(query_feat.unsqueeze(0), feat_db, similarity_metric)
            query_feat = np.dot(query_feat, np.transpose(latent_space))
            query_feat = torch.tensor(query_feat)
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
    print(f'Top {inp["m"]} image IDS after performing personalized page rank on label {inp["label"]}:')
    print(top_m_ids)
    for i,image in enumerate(top_m_imgs):
        image.save('./Outputs/'+str(top_m_ids[i])+'.jpg')

if __name__ == '__main__':
  while True: main()