## Project Phase 1

### Task 1

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

Print (in human readable form):
Options:
1. Line Chart (X-> index, Y-> 0-1 values)
2. Heatmap overlaid on image (in color moments that will be for each channel)
3. For HOG, the usual should work
4. For resnet layer3 & avgpool, should be able to just upscale and overlay mostly
5. For FC, select top-k indices, have their class names as labels on X-axis, and feature-values/probabilities on Y-axis

### Task 2

- [x] Extract features from all images and store in database

### Task 3

- [x] Decide which similarity measure to use for each feature descriptor
  - [x] Color Moment -> Pearson Correlation Coefficient
  - [x] HoG -> Intersection Similarity
  - [x] ResNet Layer 3 -> Cosine Similarity
  - [x] ResNet AvgPool -> Cosine Similarity
  - [x] ResNet FC -> Manhattan Distance

- [x] Implement similarity measures
- [x] Implement query mechanism
- [x] Implement retrieval mechanism
- [x] Implement display mechanism
---
### References

1. [Hooks: the one PyTorch trick you must know](https://tivadardanka.com/blog/hooks-the-one-pytorch-trick-you-must-know)
2. [Histogram of Oriented Gradients explained using OpenCV](https://learnopencv.com/histogram-of-oriented-gradients/)
3. [Finding the Edges (Sobel Operator) - Computerphile](https://youtu.be/uihBwtPIBxM?feature=shared)
4. [How to modify path where Torch Hub models are downloaded](https://stackoverflow.com/questions/59134499/how-to-modify-path-where-torch-hub-models-are-downloaded)

---
## Phase 2

Task List:
- [x] Task 0a: Extract features for even ids and save them with labels
- [x] Task 0b: given id, k & feat_space, visualize k similar images
- [x] task 1: given label, k & feat_space, visualize k relevant images to label
- [x] task 2a: given id/img_file, k & feat_space, list k most likely matching labels, with scores.
*Note*: input of image file is still remaining
- [ ] task 2b: given id/file & k, list k most likely matching labels, with scores using RESNET50 model
- dimensionality reduction
  - [x]: svd
  - [ ]: nnmf  - throws a non-negative value found error
  - [ ]: kmeans  - calculation of weight matrix is still remaining
  - [x]: lda  - throws a non-negative value found error
- [ ] task 3(LS1): given feat_space, k & dimension reduction (dim_red) technique, report top-k latent semanitics
  - [x]: store latent semantics
  - [ ]: list img id-weight pairs, in decreasing order of weights
- [ ] task 4(LS2): given k & feat_space, report top-k latent semanitics using CP decomposition of a 3-modal(img-feat-label) tensor
  - [ ]: store latent semantics
  - [ ]: list img id-weight pairs, in decreasing order of weights
- [ ] task 5(LS3): given feat_space, k & dimension reduction (dim_red) technique
  - [x]: create label-label similarity matrix
  - [x]: perform dim_red on above mat
  - [x]: store latent semantics
  - [ ]: list label-weight pairs, in decreasing order of weights
- [ ] task 6(LS4): given feat_space, k & dimension reduction (dim_red) technique
  - [x]: create img-img similarity matrix
  - [x]: perform dim_red on above mat
  - [x]: store latent semantics
  - [ ]: list img id-weight pairs, in decreasing order of weights
- [ ] task 7: given id/filename, latent_space & k, visualize the most similar k images
- [ ] task 8: given id/filename, latent_space & k, list k most likely matching labels, along with scores
- [ ] task 9: given label, latent_space & k, list k most likely matching labels, along with scores
- [ ] task 10: given label, latent_space & k, visualize the most similar k images
- [ ] task 11 [PageRank]: given feat/latent space, `n`, `m` & label
  - [ ] create a similaritiy graph $$G(V, E)$$, where $$V$$ corresponds to imgs in DB & draw $$E$$, if the nodes are in top `n` similar images list of each other.
  - [ ] identify mosts significant `m` images (relative to the given label) using personalized PageRank measure
