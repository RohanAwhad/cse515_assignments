import config
import helper
from similarity_metrics import get_similarity, get_top_k_ids_n_scores
import torch

def retrieve_similar_labels(l, semantic, k):
    # Load latent semantics
    latent_semantics = helper.load_pickle(config.LATENT_SEMANTICS_FN.format(task='5_label_label_simi_mat', feat_space=semantic, K=k, dim_red='svd'))
    
    # Fetch label vector for the given label 'l'
    label_vec = latent_semantics[l]
    
    # Compute similarity scores
    similarity_scores = get_similarity(label_vec, latent_semantics, 'cosine_similarity')
    
    # Get top k labels
    top_k_labels, top_k_scores = get_top_k_ids_n_scores(similarity_scores, {}, k)
    
    # Print results
    for label, score in zip(top_k_labels, top_k_scores):
        print(f"Label: {label}, Similarity Score: {score}")