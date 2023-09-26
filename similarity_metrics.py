'''
Changes made since Phase 1 submission
- created this new file and moved similarity calc funcs here
'''
import numpy as np
import torch

from numpy.linalg import norm

def cosine_similarity(query_vec, db_mat, idx_dict, K):
  query_vec = query_vec.numpy()
  similarity_scores = (query_vec @ db_mat.T) / (norm(query_vec) * norm(db_mat.T, axis=0))  # cosine similarity
  ss_arg_idx = similarity_scores.argsort()[-K:][::-1]
  top_k_scores = [similarity_scores[x] for x in ss_arg_idx]
  top_k_idx = [idx_dict[x][0] for x in ss_arg_idx]
  return top_k_idx, top_k_scores


def intersection_similarity(query_vec, db_mat, idx_dict, K):
  query_vec = query_vec.unsqueeze(0).expand(len(db_mat), -1)
  stack_ = torch.stack([query_vec, db_mat])
  similarity_scores = stack_.min(0).values.sum(-1) / stack_.max(0).values.sum(-1)
  ss_arg_idx = similarity_scores.numpy().argsort()[-K:][::-1]
  top_k_scores = [similarity_scores[x] for x in ss_arg_idx]
  top_k_idx = [idx_dict[x][0] for x in ss_arg_idx]
  return top_k_idx, top_k_scores


def pearson_coefficient(query_vec, db_mat, idx_dict, K):
  similarity_scores = torch.cat((query_vec.unsqueeze(0), 
    torch.tensor(db_mat)), dim=0).corrcoef()[0, 1:]  # pearson correlation coefficient
  ss_arg_idx = similarity_scores.numpy().argsort()[-K:][::-1]  # TODO (rohan): is conversion to numpy compulsory?
  top_k_scores = [similarity_scores[x] for x in ss_arg_idx]
  top_k_idx = [idx_dict[x][0] for x in ss_arg_idx]
  return top_k_idx, top_k_scores


def manhattan_distance(query_vec, db_mat, idx_dict, K):
  query_vec = query_vec.numpy()
  similarity_scores = np.abs(query_vec[np.newaxis, :] - db_mat).sum(-1)  # manhattan distance
  ss_arg_idx = similarity_scores.argsort()[:K]
  top_k_scores = [similarity_scores[x] for x in ss_arg_idx]
  top_k_idx = [idx_dict[x][0] for x in ss_arg_idx]
  return top_k_idx, top_k_scores

