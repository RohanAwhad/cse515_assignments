import config
import helper
import dimensionality_reduction
from feature_descriptor import FeatureDescriptor
import numpy as np
import similarity_metrics
import torch
from similarity_metrics import get_similarity, get_top_k_ids_n_scores, get_similarity_mat_x_mat
from task5 import get_label_vecs


feature_descriptor = FeatureDescriptor(net=config.RESNET_MODEL)



def main():
    labels = [label for _, label in config.DATASET]
    max_label_val = max(labels)
    inp = helper.get_user_input('task_id,K,feat_space,label,K_latent', len(config.DATASET), max_label_val)
    _tmp = config.FEAT_DESC_FUNCS[inp['feat_space']]
    feat_db, idx_dict, similarity_metric = _tmp[config.FEAT_DB], _tmp[config.IDX], _tmp[config.SIMILARITY_METRIC]
    
    # Get label feature vector
    label_feats = [x[0] for x in filter(lambda x: x[1][1] == inp['label'], idx_dict.items())]
    query_feat = torch.tensor(feat_db[label_feats, :].mean(0))
    query_image = next((image for image, label in config.DATASET if label == inp['label']), None)

    if query_image:
        # Using helper.plot() to display the query image
        helper.plot(query_image, inp['label'], [query_image], [inp['label']], [0], 1, "Query Image", "N/A")
    else:
        print("No image found for the given label:", inp['label'])
    if inp['task_id'] == 3:
        inp_2 = helper.get_user_input('dim_red', len(config.DATASET))
        inp['dim_red'] = inp_2['dim_red']
        latent_space = helper.load_semantics(f"task{inp['task_id']}_{inp['feat_space']}_{inp['dim_red']}_{inp['K_latent']}")
    
    elif inp['task_id'] == 4:
        inp['dim_red'] = "cp"
        latent_space = helper.load_semantics(f"task{inp['task_id']}_{inp['feat_space']}_{inp['dim_red']}_{inp['K_latent']}_feature.pkl")

    elif inp['task_id'] == 5:
        inp_2 = helper.get_user_input('dim_red', len(config.DATASET))
        inp['dim_red'] = inp_2['dim_red']
        latent_space = helper.load_semantics(f"task{inp['task_id']}_label_label_simi_mat_{inp['feat_space']}_{inp['dim_red']}_{inp['K_latent']}.pkl")
        feat_db_labels = get_label_vecs(inp['feat_space'])
        query_feat = get_similarity_mat_x_mat(query_feat.unsqueeze(0), feat_db_labels, similarity_metric)
        feat_db = get_similarity_mat_x_mat(feat_db, feat_db_labels, similarity_metric)
    
    elif inp['task_id'] == 6:
        inp_2 = helper.get_user_input('dim_red', len(config.DATASET))
        inp['dim_red'] = inp_2['dim_red']
        latent_space = helper.load_semantics(f"task{inp['task_id']}_img_img_simi_mat_{inp['feat_space']}_{inp['dim_red']}_{inp['K_latent']}.pkl")
        query_feat = get_similarity_mat_x_mat(query_feat.unsqueeze(0), feat_db, similarity_metric)
        feat_db = get_similarity_mat_x_mat(feat_db, feat_db, similarity_metric)

    if inp['dim_red'] == 'svd': similarity_metric = 'cosine_similarity'
    elif inp['dim_red'] == 'nnmf': similarity_metric = 'cosine_similarity'
    elif inp['dim_red'] == 'cp': similarity_metric = 'cosine_similarity'
    elif inp['dim_red'] == 'kmeans': similarity_metric = 'manhattan_distance'
    elif inp['dim_red'] == 'lda': similarity_metric = 'cosine_similarity' 
    
    train_latent_space = np.dot(feat_db, np.transpose(latent_space))
    train_latent_space = torch.tensor(train_latent_space)
    query_latent_space = np.dot(query_feat, np.transpose(latent_space))
    query_latent_space = torch.tensor(query_latent_space)

    similarity_scores = get_similarity(query_latent_space, train_latent_space, similarity_metric)
    top_k_ids, top_k_scores = get_top_k_ids_n_scores(similarity_scores, idx_dict, inp['K'])
    
    # Visualize the images
    top_k_imgs = [config.DATASET[x][0] for x in top_k_ids]
    helper.plot(None, inp['label'], top_k_imgs, top_k_ids, top_k_scores, inp['K'], inp['feat_space'], similarity_metric)

if __name__ == '__main__':
    while True:
        main()
