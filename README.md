# Project Phase 1

# Task 1

Feature Descriptors:
- [x] Color Moments
- [x] HOG[2,3].
  - [x] Map the image to gray scale
  - [x] Use $[-1, 0, 1]$ and $[-1, 0, 1]^T$ filter
  - [ ] [Optional] A small Gaussian Blur first, to remove noisy edges, which are not really edges[3] 
  - [ ] [Optional&More]Apply, Sobel filter to get `grad_x` & `grad_y`
  - [x] Calculate magnitude and direction of gradients
  - [x] Compute 9-bin (signed) magnitude-weighted gradient histogram, with each bin corresponding to 40 degrees.
- [x] ResNet: Used hooks to save intermediate output[1].
  - [x] ResNet AvgPool
  - [x] ResNet Layer 3
  - [x] ResNet FC

---
# References

1. [Hooks: the one PyTorch trick you must know](https://tivadardanka.com/blog/hooks-the-one-pytorch-trick-you-must-know)
2. [Histogram of Oriented Gradients explained using OpenCV](https://learnopencv.com/histogram-of-oriented-gradients/)
3. [Finding the Edges (Sobel Operator) - Computerphile](https://youtu.be/uihBwtPIBxM?feature=shared)
4. [How to modify path where Torch Hub models are downloaded](https://stackoverflow.com/questions/59134499/how-to-modify-path-where-torch-hub-models-are-downloaded)
