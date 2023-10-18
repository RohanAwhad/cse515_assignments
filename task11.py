import config
import helper

from config import FEAT_DESC_FUNCS
from similarity_metrics import get_similarity_mat_x_mat

from task5 import get_label_vecs
from task6 import get_img_img_similarity_matrix

import torch
from tqdm import tqdm


'''
feat_space = 'resnet_softmax'
n = 10
m = 5
label = 100
'''

def main():
    inp = helper.get_user_input('LORF,feat_space,n,m,label,alpha',len(config.DATASET), len(set(config.DATASET.y)))
    _tmp = config.FEAT_DESC_FUNCS[inp['feat_space']]


    if inp['LORF'] == 1:
        similarity_metric = _tmp[config.SIMILARITY_METRIC]
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
        
        feat_db = feat_temp @ latent_space.T
        similarity_metric = 'cosine_similarity'  # TODO (rohan): should be based on the dimension reductionality method
        feat_similarity_scores = get_similarity_mat_x_mat(feat_db, feat_db, similarity_metric)

    else:
        similarity_metric = _tmp[config.SIMILARITY_METRIC]
        feat_similarity_scores = get_img_img_similarity_matrix(inp['feat_space'])

    # Task 1:
    # – creates a similarity graph, G(V, E), where V corresponds to the images in the database and E contains node pairs vi , vj such that, for each subject vi , vj is one of the n most similar images in the database in the given space

    # Use config and load image_feature matrix for featuer space 'resnet_softmax'
    # Calculate similarity matrix using img-img similarity matrix function from task6
    # Argsort for each row and get top n indices
    # Create an adjacency matrix with 1s for top n indices and 0s for rest for each image

    # Task 2:
    # - identifies the most significant m images (relative to the given label l) using personalized PageRank measure.

    # Create an array of len(nodes)
    # Set all to 0
    # Set images belonging to label l to 1
    # Calculate personalized page rank for each image
    # Sort and get top m images

    if similarity_metric in config.DISTANCE_MEASURES:
        _, similar_img_ids = torch.topk(feat_similarity_scores, inp['n'])
    else:
        _, similar_img_ids = torch.topk((feat_similarity_scores), inp['n'])

    print(similarity_metric)
    print(similar_img_ids[0])
    print(feat_similarity_scores[0][similar_img_ids[0]])
    print(similar_img_ids.shape)

    # create adjacency matrix

    adj_mat = torch.zeros((len(similar_img_ids), len(similar_img_ids)))
    for i in range(len(similar_img_ids)):
        adj_mat[i][similar_img_ids[i]] = 1
    print(adj_mat[0].sum())

    # Task 2

    personalization = torch.zeros((len(similar_img_ids)))
    # for images belonging to label l, set personalization to 1
    # _tmp[config.IDX] is idx to (img_id, label)
    # I want a label to list of idx
    query_img = None
    ids_to_set_to_1 = []
    idx_dict = _tmp[config.IDX]
    for x, (img_id, img_l) in idx_dict.items():
        if img_l == inp['label']:
            ids_to_set_to_1.append(x)
            if query_img is None:
                query_img = (config.DATASET[img_id][0], img_l)

    personalization[ids_to_set_to_1] = 1
    # convert it to probability distribution
    personalization = personalization / personalization.sum()

    print(personalization.sum())

    def compute_transition_matrix(adj_matrix):
        row_sum = torch.sum(adj_matrix, dim=1, keepdim=True)
        return adj_matrix / row_sum

    def personalized_page_rank(adj_matrix, personalization, alpha=0.85, max_iter=100, tol=1e-6):
        # Get the transition matrix
        M = compute_transition_matrix(adj_matrix)
        
        # Number of nodes
        N = adj_matrix.shape[0]
        
        # Initialize PageRank vector
        R = torch.ones(N) / N
        
        # Compute the teleporting vector based on personalization
        S = personalization / torch.sum(personalization)
        
        for _ in tqdm(range(max_iter)):
            R_next = alpha * torch.mv(M.t(), R) + (1 - alpha) * S
            
            # Check for convergence
            if torch.norm(R_next - R, 1) <= tol:
                break
                
            R = R_next
            
        return R

    page_rank = personalized_page_rank(adj_mat, personalization, inp['alpha'])
    print(page_rank)
    print(page_rank.shape)

    # Sort and get top m images
    top_m_values, top_m_ids = torch.topk(page_rank, inp['m'])
    # convert to list
    top_m_ids = top_m_ids.tolist()
    top_m_values = top_m_values.tolist()
    print(list(zip(top_m_ids, top_m_values)))

    # get top m images
    top_m_ids = [idx_dict[x][0] for x in top_m_ids]
    top_m_imgs = [config.DATASET[x][0] for x in top_m_ids]

    helper.plot(
        query_img[0],
        query_img[1],
        top_m_imgs,
        top_m_ids,
        top_m_values,
        K=inp['m'],
        row_label=inp['feat_space'],
        similarity_function='Personalized Page Rank'
    )

if __name__ == '__main__':
  while True: main()