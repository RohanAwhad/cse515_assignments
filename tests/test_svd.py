from dimensionality_reduction import svd_custom
import torch
from numpy.linalg import svd

def test_svd_custom():
  # create a random mat 5x7 matrix
  mat = torch.randn(5, 7)
  # compute the svd
  U_sigma, V_T = svd_custom(mat)

  exp_u, exp_sigma, exp_v = svd(mat.numpy())
  exp_u_sigma = torch.tensor(exp_u @ exp_sigma)

  print()
  print("Expected")
  print("Sigma: ", exp_sigma)
  print("U    : ", exp_u)
  print("V    : ", exp_v)

  print("U_sigma    : ", U_sigma)
  print("exp_u_sigma: ", exp_u_sigma)
  torch.testing.assert_allclose(U_sigma, exp_u_sigma)
  torch.testing.assert_allclose(V_T, torch.tensor(exp_v))


