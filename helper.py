'''
Changes since Phase 1 Submission:
- added a plot function to show task0a results during runtime
'''
import matplotlib.pyplot as plt
import pickle
import bz2

def plot(img, query_img_id, top_k_imgs, top_k_ids, top_k_img_scores, K, row_label, similarity_function):
  n_rows = 2
  fig, axes = plt.subplots(n_rows, K, figsize=(K*2, n_rows*2))
  axes[0, 0].imshow(img)
  axes[0, 0].set_xlabel(f'Img ID: {query_img_id}')
  axes[0, 0].set_xticks([])
  axes[0, 0].set_yticks([])
  for i in range(1, K): axes[0, i].axis('off')

  for i, (img, idx, score) in enumerate(zip(top_k_imgs, top_k_ids, top_k_img_scores)):
    ax = axes[1, i]
    ax.set_xlabel(f'Img ID: {idx}\nScore/Distance: {score:0.2f}')
    if img.mode == 'L': img = img.convert('RGB')
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])

  axes[1, 0].set_ylabel(similarity_function)
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
  with open(f'features/{fd}.bin', 'wb') as f: f.write(compressed_bin_file)

def load_data(fd):
  with open(f'features/{fd}.bin', 'rb') as f: cmprsd_bin = f.read()
  bin_data = bz2.decompress(cmprsd_bin)
  return pickle.loads(bin_data)
