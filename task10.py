def retrieve_similar_images(l, semantic, k):
    # Load latent semantics
    label_latent_semantics = helper.load_pickle(config.LATENT_SEMANTICS_FN.format(task='5_label_label_simi_mat', feat_space=semantic, K=k, dim_red='svd'))
    img_latent_semantics = helper.load_pickle(config.LATENT_SEMANTICS_FN.format(task='6_img_img_simi_mat', feat_space=semantic, K=k, dim_red='svd'))
    
    # Fetch label vector for the given label 'l'
    label_vec = label_latent_semantics[l]
    
    # Compute similarity scores
    similarity_scores = get_similarity(label_vec, img_latent_semantics, 'cosine_similarity')
    
    # Get top k images
    top_k_img_ids, top_k_scores = get_top_k_ids_n_scores(similarity_scores, config.FEAT_DESC_FUNCS[semantic][config.IDX], k)
    
    # Print results
    for img_id, score in zip(top_k_img_ids, top_k_scores):
        print(f"Image ID: {img_id}, Similarity Score: {score}")
