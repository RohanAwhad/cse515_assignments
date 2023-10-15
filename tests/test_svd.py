from dimensionality_reduction import svd_custom, nnmf_custom
import torch
from numpy.linalg import svd
from sklearn.decomposition import NMF

import numpy as np
np.set_printoptions(precision=3, suppress=True)

def test_svd_custom():
  # create a random mat 5x7 matrix
  mat = torch.randn(5, 7)
  # compute the svd
  U_sigma, V_T = svd_custom(mat, 2)

  exp_u, exp_sigma, exp_v = svd(mat.numpy())
  exp_u_sigma = torch.tensor(exp_u @ exp_sigma)

  print()
  print("Expected")
  print("Sigma: ", exp_sigma)
  print("U    : ", exp_u)
  print("V    : ", exp_v)

  print("U_sigma    : ", U_sigma)
  print("exp_u_sigma: ", exp_u_sigma)
  assert False


def test_nnmf_custom():
  mat = torch.randn(5, 7).abs()
  model = NMF(n_components=2, init='random', solver='mu')
  W = model.fit_transform(mat)
  H = model.components_

  W_custom, H_custom = nnmf_custom(mat, 2, max_iter=200)

  print()
  print("Expected")
  print("W: ", W)
  print("H: ", H)

  print()
  print("Actual")
  print("W: ", W_custom.numpy())
  print("H: ", H_custom.numpy())

  assert False
