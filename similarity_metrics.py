'''
Changes made since Phase 1 submission
- created this new file and moved similarity calc funcs here
'''
import numpy as np
import torch

from torch.linalg import norm
from tqdm import tqdm
from typing import Dict, Tuple, List

def get_top_k_ids_n_scores(
  similarity_scores: torch.Tensor, idx_dict: Dict[int, Tuple[int, int]], K: int
) -> Tuple[List[int], List[float]]:

  ss_arg_idx = similarity_scores.argsort()[-K:][::-1]
  top_k_scores = [similarity_scores[x].item() for x in ss_arg_idx]
  top_k_idx = [idx_dict[x][0] for x in ss_arg_idx]
  return top_k_idx, top_k_scores

def get_similarity(query_vec: torch.Tensor, db_mat: torch.Tensor, similarity_metric: str) -> torch.Tensor:
  if similarity_metric == 'cosine_similarity':
    return (query_vec @ db_mat.T) / (norm(query_vec) * norm(db_mat.T, axis=0))

  elif similarity_metric == 'intersection_similarity':
    query_vec = query_vec.unsqueeze(0).expand(len(db_mat), -1)
    stack_ = torch.stack([query_vec, db_mat])
    return stack_.min(0).values.sum(-1) / stack_.max(0).values.sum(-1)

  elif similarity_metric == 'pearson_coefficient':
    return torch.cat((query_vec.unsqueeze(0), db_mat), dim=0).corrcoef()[0, 1:]

  elif similarity_metric == 'manhattan_distance':
    if len(query_vec.shape) < 2: query_vec = query_vec.unsqueeze(0)
    return -1 * torch.abs(query_vec - db_mat).sum(-1)  # multiply by -1 for retrieving top k functionality

  else:
    raise NotImplementedError(f'{similarity_metric} algorithm not implemented')


def get_similarity_mat_x_mat(mat1: torch.Tensor, mat2: torch.Tensor, similarity_metric: str) -> torch.Tensor:
  if similarity_metric == 'cosine_similarity':
    print('calculating similarity ...')
    return get_similarity(mat1, mat2, similarity_metric)
  return torch.stack([
    get_similarity(row, mat2, similarity_metric) for row in tqdm(mat1, desc='Calculating similarity')
  ])
