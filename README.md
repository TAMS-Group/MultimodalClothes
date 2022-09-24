# MultimodalClothes

## Data

The dataset itself is available to download as a zip file via https://tams-www.informatik.uni-hamburg.de/download/MultimodalClothes.zip.
The compressed file has a size of 117GB. 

We provide multimodal recordings of pieces of clothing grasped and rotated by a robot arm. 
![](img/modalities.png)

### Statistics

![](img/sample_count.png)

## Code 

In the code directory, all of the code used for training, testing and data selection and preprocessing can be found. 
The code used in this work is a heavily modified version of the following repository: https://github.com/yanx27/Pointnet_Pointnet2_pytorch .
We thank the contributors for publishing their work.

## Benchmark performance

We evaluate common architectures (the code of which is included in this repository) on our dataset with the following results: 

![](img/benchmark_performance.png)
