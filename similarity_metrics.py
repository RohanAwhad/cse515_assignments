'''
Changes made since Phase 1 submission
- created this new file and moved similarity calc funcs here
'''
import numpy as np
import torch

from numpy.linalg import norm

to_tensor = lambda x: torch.tensor(x)

def get_top_k_ids_n_scores(similarity_scores, idx_dict, K):
  ss_arg_idx = similarity_scores.argsort()[-K:][::-1]
  top_k_scores = [similarity_scores[x] for x in ss_arg_idx]
  top_k_idx = [idx_dict[x][0] for x in ss_arg_idx]
  return top_k_idx, top_k_scores

def get_similarity(query_vec, db_mat, similarity_metric: str):
  print(f'query_vec size: {query_vec.shape} | db_mat size: {db_mat.shape}')
  print(f'query_vec type: {type(query_vec)} | db_mat type: {type(db_mat)}')
  if similarity_metric == 'cosine_similarity':
    if not isinstance(query_vec, np.ndarray): query_vec.numpy()
    similarity_scores = (query_vec @ db_mat.T) / (norm(query_vec) * norm(db_mat.T, axis=0))  # cosine similarity

  elif similarity_metric == 'intersection_similarity':
    if not isinstance(query_vec, torch.Tensor): query_vec = torch.tensor(query_vec)
    query_vec = query_vec.unsqueeze(0).expand(len(db_mat), -1)
    stack_ = torch.stack([to_tensor(query_vec), to_tensor(db_mat)])
    similarity_scores = stack_.min(0).values.sum(-1) / stack_.max(0).values.sum(-1)

  elif similarity_metric == 'pearson_coefficient':
    if not isinstance(query_vec, torch.Tensor): query_vec = torch.tensor(query_vec)
    similarity_scores = torch.cat((query_vec.unsqueeze(0), 
      torch.tensor(db_mat)), dim=0).corrcoef()[0, 1:]  # pearson correlation coefficient

  elif similarity_metric == 'manhattan_distance':
    if not isinstance(query_vec, np.ndarray): query_vec.numpy()
    if len(query_vec.shape) < 2: query_vec = query_vec[np.newaxis, :]
    similarity_scores = -1 * np.abs(query_vec - db_mat).sum(-1)  # manhattan distance; multiply by -1 for retrieving top k functionality

  else:
    raise NotImplementedError(f'{similarity_metric} algorithm not implemented')

  return similarity_scores


def get_similarity_mat_x_mat(mat1, mat2, similarity_metric):
  if similarity_metric in ('pearson_coefficient', 'intersection_similarity'):
    return torch.stack([get_similarity(row, mat2, similarity_metric) for row in mat1])

  else: return get_similarity(mat1, mat2, similarity_metric)

    
