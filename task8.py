#!python3

import config
import helper
import dimensionality_reduction
from feature_descriptor import FeatureDescriptor
import numpy as np
import similarity_metrics
import torch
from similarity_metrics import get_similarity, get_top_k_ids_n_scores, get_similarity_mat_x_mat
from task2a import get_label_vecs

feature_descriptor = FeatureDescriptor(net=config.RESNET_MODEL)

def get_label_vecs(feat_space: str):
  ret = []
  for label in range(max(config.DATASET.y) + 1):
    # get images feats for the given label
    *_, feat_db_idx, feat_db = config.FEAT_DESC_FUNCS[feat_space]
    label_feats = [x[0] for x in filter(lambda x: x[1][1] == label, feat_db_idx.items())]

    # pool feat_matrix into vector
    # TODO (rohan): Decide a better pooling method
    label_feat = torch.tensor(feat_db[label_feats, :].mean(0))
    ret.append(label_feat)

  return torch.stack(ret, dim=0)
  

def main():
  
  inp = helper.get_user_input('task_id,K,feat_space,img_id,K_latent',len(config.DATASET))
  _tmp = config.FEAT_DESC_FUNCS[inp['feat_space']]
  feat_db, idx_dict, similarity_metric = _tmp[config.FEAT_DB], _tmp[config.IDX], _tmp[config.SIMILARITY_METRIC]
  feat_db_labels = get_label_vecs(inp['feat_space'])
  if isinstance(inp['img_id'], str): img = helper.load_img_file(inp['img_id'])
  else: img = config.DATASET[inp['img_id']][0]
  if img.mode != 'RGB': img = img.convert('RGB')  

  if(inp['task_id']==3):
    inp_2 = helper.get_user_input('dim_red',len(config.DATASET))
    inp['dim_red'] = inp_2['dim_red']
    latent_space = helper.load_semantics("task" + str(inp['task_id'])+ "_" + inp['feat_space']+"_"+inp['dim_red']+"_"+str(inp['K_latent'])+".pkl")
    query_feat = feature_descriptor.extract_features(img, inp['feat_space'])


  elif(inp['task_id']==4):
    inp['dim_red']="cp"
    latent_space = helper.load_semantics("task" + str(inp['task_id'])+ "_" + inp['feat_space']+"_"+inp['dim_red']+"_"+str(inp['K_latent'])+"_feature.pkl")
    query_feat = feature_descriptor.extract_features(img, inp['feat_space'])
    

  elif(inp['task_id']==5):
    inp_2 = helper.get_user_input('dim_red',len(config.DATASET))
    inp['dim_red'] = inp_2['dim_red']
    latent_space = helper.load_semantics("task" + str(inp['task_id']) + "_label_label_simi_mat"+ "_" + inp['feat_space']+"_"+inp['dim_red']+"_"+str(inp['K_latent'])+".pkl")
    query_feat = feature_descriptor.extract_features(img, inp['feat_space'])
    query_feat = get_similarity_mat_x_mat(query_feat.unsqueeze(0), feat_db_labels, similarity_metric)
    feat_db_labels = get_similarity_mat_x_mat(feat_db_labels, feat_db_labels, similarity_metric)

  elif(inp['task_id']==6):
    
    inp_2 = helper.get_user_input('dim_red',len(config.DATASET))
    inp['dim_red'] = inp_2['dim_red']
    latent_space = helper.load_semantics("task" + str(inp['task_id']) + "_img_img_simi_mat"+ "_" + inp['feat_space']+"_"+inp['dim_red']+"_"+str(inp['K_latent'])+".pkl")
    query_feat = feature_descriptor.extract_features(img, inp['feat_space'])
    query_feat = get_similarity_mat_x_mat(query_feat.unsqueeze(0), feat_db, similarity_metric)
    feat_db_labels = get_similarity_mat_x_mat(feat_db_labels, feat_db, similarity_metric)
    

  train_latent_space = np.dot(feat_db_labels, np.transpose(latent_space))
  train_latent_space = torch.tensor(train_latent_space)
  query_latent_space = np.dot(query_feat, np.transpose(latent_space))
  query_latent_space = torch.tensor(query_latent_space)
  similarity_scores = get_similarity(query_latent_space, train_latent_space, similarity_metric)
  top_k_ids, top_k_scores = get_top_k_ids_n_scores(similarity_scores, idx_dict, inp['K'])
  # print output
  img_id = inp['img_id']
  print('-'*50)
  print(f'Labels similar to img_id: {img_id}')
  for idx, score in zip(top_k_ids, top_k_scores): print(f'  - Label: {idx} | Score: {score}')
  print(f'\n  - Original Label: {config.DATASET[img_id][1]}')
  print('-'*50)
  

if __name__ == '__main__':
  while True: main()

