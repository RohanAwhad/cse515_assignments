'''
Changes made since Phase 1 submission
- created this new file and moved similarity calc funcs here
'''
from config import DEBUG

import numpy as np
import queue
import sys
import threading
import torch

from torch.linalg import norm
from tqdm import tqdm
from typing import Dict, Tuple, List

# TODO (rohan): refactor debug statements

def get_top_k_ids_n_scores(
  similarity_scores: torch.Tensor, idx_dict: Dict[int, Tuple[int, int]], K: int
) -> Tuple[List[int], List[float]]:

  ss_arg_idx = reversed(similarity_scores.argsort()[-K:])
  top_k_scores = [similarity_scores[x].item() for x in ss_arg_idx]
  top_k_idx = [idx_dict[x.item()][0] for x in ss_arg_idx]
  return top_k_idx, top_k_scores

def get_similarity(query_vec: torch.Tensor, db_mat: torch.Tensor, similarity_metric: str) -> torch.Tensor:
  if similarity_metric == 'cosine_similarity':
    ret = (query_vec @ db_mat.T) / (norm(query_vec) * norm(db_mat.T, axis=0))

  elif similarity_metric == 'intersection_similarity':
    query_vec = query_vec.unsqueeze(0).expand(len(db_mat), -1)
    stack_ = torch.stack([query_vec, db_mat])
    ret = stack_.min(0).values.sum(-1) / stack_.max(0).values.sum(-1)

  elif similarity_metric == 'pearson_coefficient':
    ret = torch.cat((query_vec.unsqueeze(0), db_mat), dim=0).corrcoef()[0, 1:]
    # there is a lot of overhead in this tensor. Don't know why. to overcome this
    # converting to numpy and then back to tensor.
    ret = torch.tensor(ret.numpy())

  elif similarity_metric == 'manhattan_distance':
    if len(query_vec.shape) < 2: query_vec = query_vec.unsqueeze(0)
    ret = -1 * torch.abs(query_vec - db_mat).sum(-1)  # multiply by -1 for retrieving top k functionality

  elif similarity_metric == 'kl_divergence':
    if len(query_vec.shape) < 2: query_vec = query_vec.unsqueeze(0)
    ret = -1 * (query_vec * (query_vec / db_mat).log()).sum(-1)  # returns 1D tensor

  else:
    raise NotImplementedError(f'{similarity_metric} algorithm not implemented')

  if DEBUG>1:
    total_size_tensor = sys.getsizeof(ret.storage())
    actual_elements_size = ret.element_size() * ret.nelement()
    overhead = total_size_tensor - actual_elements_size
    tmp = ret.tolist()
    element_size = sys.getsizeof(tmp[0])
    total_list_size = element_size * len(tmp)
    print(f'size of similarity scores tensor\n\
    - {"Total :":20s}{total_size_tensor/1e6:10.2f} MB\n\
    - {"Actual elements :":20s}{actual_elements_size/1e6:10.2f} MB\n\
    - {"OverHead :":20s}{overhead/1e6:10.2f} MB\n\
    - {"Shape :":20s}{ret.size()}\n\
    - {"total elements :":20s}{ret.nelement()}\n\
size of the same tensor in list format:{total_list_size/1e6:10.2f} MB')
  return ret


def get_similarity_mat_x_mat(mat1: torch.Tensor, mat2: torch.Tensor, similarity_metric: str) -> torch.Tensor:
  if similarity_metric == 'cosine_similarity':
    print('calculating similarity ...')
    return get_similarity(mat1, mat2, similarity_metric)

  elif similarity_metric == 'pearson_coefficient':
    return torch.cat((mat1, mat2), dim=0).corrcoef()[:len(mat1), len(mat1):]

  # implementing multithreading
  def thread_func(row, mat2, similarity_metric, ret, tid):
    if DEBUG>1: print(f'func_name: thread_func\t\ttid: {tid}')
    ret[tid] = get_similarity(row, mat2, similarity_metric)
    if DEBUG>1: print(f'func_name: thread_func\t\ttid: {tid}\t\t DONE')
    

  _tmp = [None]*len(mat1)
  # utilizes too much ram by initializing a ton of threads. Need a pool executor
  threads = queue.Queue()
  max_threads = 50
  prog_bar = tqdm(desc='Calculating similarity', total=len(mat1))
  for i, row in enumerate(mat1):
    while threads.qsize() >= max_threads:
      _i = range(threads.qsize())
      for _ in _i:
        x = threads.get()
        if x.is_alive(): threads.put(x)
        else: prog_bar.update(1)

    x = threading.Thread(target=thread_func, args=(row, mat2, similarity_metric, _tmp, i,))
    threads.put(x)
    x.start()

  while not threads.empty(): 
    t = threads.get()
    t.join()
    prog_bar.update(1)

  return torch.stack(_tmp)

