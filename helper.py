'''
Changes since Phase 1 Submission:
- added a plot function to show task0a results during runtime
'''

# python libs
import bz2
import os
import pickle

from typing import Dict, Union, Tuple, Any

# 3rd-party libs
import matplotlib.pyplot as plt

from PIL import Image
import os

def load_img_file(fn):
  return Image.open(fn)

def get_user_input(inp: str, len_ds: int=0, max_label_val: int=0) -> Dict[str, Union[str, int]]:
  feature_descriptor_dict: Dict[int, str] = {
    1: 'color_moment',
    2: 'hog',
    3: 'resnet_layer3',
    4: 'resnet_avgpool',
    5: 'resnet_fc',
  }
  dim_red_dict: Dict[int, str] = {
    1: 'svd',
    2: 'nnmf',
    3: 'kmeans',
    4: 'lda',
  }
  latent_semantic_dict: Dict[int, str] = {
    3: 'svd',
    4: 'nnmf',
    5: 'kmeans',
    6: 'lda',
  }

  ret: Dict[str, Union[str, int]] = {}
  for x in inp.split(','):
    try:
      if x == 'K':
        ret[x] = int(input('Input K for top-k: '))
        if ret[x] == 0: raise ValueError(f'K needs to be greater than 0. Given: {ret[x]}')
      
      elif x == 'K_latent':
        ret[x] = int(input('Input K for latent space: '))
        if ret[x] == 0: raise ValueError(f'K needs to be greater than 0. Given: {ret[x]}')

      elif x == 'img_id':
        is_img_fn = input('Do you want to enter image filename? (y/N): ').lower()
        if is_img_fn == 'y':
          ret[x] = input('Enter filename: ')
        else:
          ret[x] = int(input(f'Enter an image id [0, {len_ds-1}]: '))
          if ret[x] < 0 or ret[x] >= len_ds:
            raise ValueError(f'img id invalid. should be between [0, {len_ds-1}], try again. you got it!')

      elif x == 'feat_space':
        fd_id = int(input('''Enter id of feature descriptor

1: color_moment
2: hog
3: resnet_layer3
4: resnet_avgpool
5: resnet_fc

>'''
        ))
        if not (0 < fd_id < 6): raise ValueError('value should be between [1, 5]')
        ret[x] = feature_descriptor_dict[fd_id]

      elif x == 'task_id':
        task_id = int(input('''Enter id of task

3: LS1
4: LS2
5: LS3
6: LS4  
>'''
        ))
        if not (3 <= task_id <= 6): raise ValueError('value should be between [3, 6]')
        ret[x] = task_id

      elif x == 'label':
        ret[x] = int(input(f'Enter a query label [0, {max_label_val}]: '))
        if ret[x] < 0 or ret[x] > max_label_val:
          raise ValueError(f'label invalid. should be between [0, {max_label_val}], try again. you got it!')
      elif x == 'dim_red':
        _id = int(input('''Enter id of dimension reductionality algorithm to use:

1: svd
2: nnmf
3: kmeans
4: lda

>'''
        ))
        if not (0 < _id < 5): raise ValueError('value should be between [1, 4]')
        ret[x] = dim_red_dict[_id]

      else:
        raise ValueError(f'{x} not yet implemented')


    except KeyboardInterrupt:
      print('\nBye bye ...')
      exit(0)

  return ret

def plot(img, query_img_id, top_k_imgs, top_k_ids, top_k_img_scores, K, row_label, similarity_function):
  n_rows = 1 if img is None else 2
  fig, axes = plt.subplots(n_rows, K, figsize=(K*2, n_rows*2))

  if K == 1:
    if img is not None:
      axes[0].imshow(img)
      axes[0].set_xlabel(f'Img ID: {query_img_id}')
      axes[0].set_xticks([])
      axes[0].set_yticks([])

    img = top_k_imgs[0]
    idx = top_k_ids[0]
    score = top_k_img_scores[0]

    ax = axes[1]
    ax.set_xlabel(f'Img ID: {idx}\nScore/Distance: {score:0.2f}')
    if img.mode == 'L': img = img.convert('RGB')
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])

    if n_rows == 2: ax.set_ylabel(similarity_function)
    else: axes[0].set_ylabel(similarity_function)

  else:
    if img is not None:
      axes[0, 0].imshow(img)
      axes[0, 0].set_xlabel(f'Img ID: {query_img_id}')
      axes[0, 0].set_xticks([])
      axes[0, 0].set_yticks([])
      for i in range(1, K): axes[0, i].axis('off')

    for i, (img, idx, score) in enumerate(zip(top_k_imgs, top_k_ids, top_k_img_scores)):
      ax = axes[i] if n_rows == 1 else axes[1, i]
      ax.set_xlabel(f'Img ID: {idx}\nScore/Distance: {score:0.2f}')
      if img.mode == 'L': img = img.convert('RGB')
      ax.imshow(img)
      ax.set_xticks([])
      ax.set_yticks([])

    if n_rows == 2: axes[1, 0].set_ylabel(similarity_function)
    else: axes[0].set_ylabel(similarity_function)

  fig.suptitle(row_label)
  plt.tight_layout()
  plt.show()

def save_top_k(img, query_img_id, top_k_imgs, top_k_ids, K, fn):
  n_rows = len(top_k_imgs)+1
  fig, axes = plt.subplots(n_rows, K, figsize=(K*2, n_rows*2))
  axes[0, 0].imshow(img)
  axes[0, 0].set_xlabel(f'Img ID: {query_img_id}')
  axes[0, 0].set_xticks([])
  axes[0, 0].set_yticks([])
  for i in range(1, K): axes[0, i].axis('off')
  #for ax, img in zip(axes[1], top_k_imgs):
  for j, (feat_imgs, feat_ids) in enumerate(zip(top_k_imgs, top_k_ids)):
    for i, (img, idx) in enumerate(zip(feat_imgs, feat_ids)):
      ax = axes[j+1, i]
      ax.set_xlabel(f'Img ID: {idx}')
      if img.mode == 'L': img = img.convert('RGB')
      ax.imshow(img)
      ax.set_xticks([])
      ax.set_yticks([])

  row_labels = ['Query Image', 'Color Moment', 'HOG', 'ResNet AvgPool', 'ResNet Layer3', 'ResNet FC']
  for i, label in enumerate(row_labels):
    axes[i, 0].set_ylabel(label)

  plt.tight_layout()
  plt.savefig(fn)

def save_data(data_tuple, fd):
  binary_file = pickle.dumps(data_tuple)
  compressed_bin_file = bz2.compress(binary_file)
  print(fd)
  print(f'- orginal size: {len(binary_file) / 1e6} MB')
  print(f'- compressed size: {len(compressed_bin_file) / 1e6} MB')
  if not os.path.exists("features/"):
    os.makedirs('features/')
     
  with open(f'features/{fd}.bin', 'wb') as f: f.write(compressed_bin_file)

def load_data(fd: str) -> Tuple[Any, ...]:
  data_fn = f'features/{fd}.bin'
  if not os.path.exists(data_fn):
    return (None, None)
    raise ValueError(f'\'{data_fn}\' does not exist')
  with open(data_fn, 'rb') as f: cmprsd_bin = f.read()
  bin_data = bz2.decompress(cmprsd_bin)
  return pickle.loads(bin_data)

def load_semantics(fd: str) -> Tuple[Any, ...]:
  with open(f'latent_semantics/{fd}', 'rb') as f: data = pickle.load(f)
  return data

def save_pickle(obj, fn):
  with open(fn, 'wb') as f: pickle.dump(obj, f)

def load_pickle(fn):
  with open(fn, 'rb') as f: return pickle.load(f)
