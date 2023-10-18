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
    latent_space = helper.load_pickle(config.LATENT_SEMANTICS_FN.format(
      task=inp['task_id'],
      feat_space=inp['feat_space'],
      dim_red=inp['dim_red'],
      K=inp['K_latent']
    ))
    query_feat = feature_descriptor.extract_features(img, inp['feat_space'])


  elif(inp['task_id']==4):
    inp['dim_red']="cp"
    latent_space = helper.load_pickle(config.LATENT_SEMANTICS_FN.format(
      task=inp['task_id'],
      feat_space=inp['feat_space'],
      dim_red=inp['dim_red'],
      K=inp['K_latent'],
      mode='feature'
    ))
    query_feat = feature_descriptor.extract_features(img, inp['feat_space'])
    

  elif(inp['task_id']==5):
    inp_2 = helper.get_user_input('dim_red',len(config.DATASET))
    inp['dim_red'] = inp_2['dim_red']
    latent_space = helper.load_pickle(config.LATENT_SEMANTICS_FN.format(
      task=inp['task_id'],
      feat_space='label_label_simi_mat_' + inp['feat_space'],
      dim_red=inp['dim_red'],
      K=inp['K_latent']
    ))
    query_feat = feature_descriptor.extract_features(img, inp['feat_space'])
    query_feat = get_similarity(query_feat, feat_db_labels, similarity_metric)
    feat_db_labels = get_similarity_mat_x_mat(feat_db_labels, feat_db_labels, similarity_metric)

  elif(inp['task_id']==6):
    # (TODO): Errors for this task
    
    inp_2 = helper.get_user_input('dim_red',len(config.DATASET))
    inp['dim_red'] = inp_2['dim_red']
    latent_space = helper.load_pickle(config.LATENT_SEMANTICS_FN.format(
      task=inp['task_id'],
      feat_space='img_img_simi_mat_' + inp['feat_space'],
      dim_red=inp['dim_red'],
      K=inp['K_latent']
    ))
    query_feat = feature_descriptor.extract_features(img, inp['feat_space'])
    query_feat = get_similarity(query_feat, feat_db, similarity_metric)
    feat_db_labels = get_similarity_mat_x_mat(feat_db_labels, feat_db, similarity_metric)

  if inp['dim_red'] == 'svd': similarity_metric = 'cosine_similarity'
  elif inp['dim_red'] == 'nnmf': similarity_metric = 'cosine_similarity'
  elif inp['dim_red'] == 'cp': similarity_metric = 'cosine_similarity'
  elif inp['dim_red'] == 'kmeans': similarity_metric = 'manhattan_distance'
  elif inp['dim_red'] == 'lda': similarity_metric = 'cosine_similarity' 

  # TODO (rohan): handle transformation specifically for Kmeans
  train_latent_space = np.dot(feat_db_labels, np.transpose(latent_space))
  train_latent_space = torch.tensor(train_latent_space)
  query_latent_space = np.dot(query_feat, np.transpose(latent_space))
  query_latent_space = torch.tensor(query_latent_space)
  similarity_scores = get_similarity(query_latent_space, train_latent_space, similarity_metric)
  top_k_ids, top_k_scores = get_top_k_ids_n_scores(similarity_scores, idx_dict, inp['K'])

  #if similarity_metric in config.DISTANCE_MEASURES: top_k_scores = list(map(abs, top_k_scores))

  # print output
  img_id = inp['img_id']
  print('-'*50)
  print(f'Labels similar to img_id: {img_id}')
  for idx, score in zip(top_k_ids, top_k_scores): print(f'  - Label: {idx:3d} | Score: {score:10.3f}')
  print(f'\n  - Original Label: {config.DATASET[img_id][1]:3d}')
  print('-'*50)
  

if __name__ == '__main__':
  while True: main()

