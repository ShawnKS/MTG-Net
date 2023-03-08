## MTG-Net

The repository contains the codes for the paper, Efficient and Effective Multi-task Grouping via Meta Learning on Task Combinations, which is accepted by Advances in Neural Information Processing Systems (NeurIPS) 2022.

#### Dataset & Backbone Model
Taskonomy We follow the backbone model used in [Which Tasks to Train Together in Multi-Task Learning](https://github.com/tstandley/taskgrouping) (ICML'20)
MIMICIII We follow the backbone model used in [A Comprehensive EHR Timeseries Pre-training Benchmark](https://dl.acm.org/doi/pdf/10.1145/3450439.3451877) (CHIL'21)
ETTm1 We follow the backbone model used in [Autoformer](https://github.com/thuml/Autoformer) (NeurIPS21)

#### Transferring Gain Collection

We provide scripts for collecting the transferring gain [here](./gain_collection).

##### Env
torch 1.7.1
cuda 10.2
numpy 1.21.6
pandas 1.3.5

For the backbone environment package used, please refer to the link mentioned above.
##### Baselines
High Order Approximation(HOA) uses pairwise task transferring gains to predict tasks' transferring gains, 
Task Affinity Grouping (TAG) combines HOA with task affinity to accelerate the data collection process.
For each baseline we provide scripts for data collect and predict.

### Pipeline

1. Collect ground truth transferring gains on datasets, our collected gains can be found at `./gain_collection/`
2. Run MTG-Net script to obtain the predicted gain, the log files would be generated at `./log/`.

A script `embed_visual_example.ipynb` to visualize the task embedding is provided.
```
python MTG.py --dataset mimic27 --gpu_id 0 --layer_num 4 --seed 72 --strategy active --num_hidden 64
```

3. We provide grouping and visualization notebooks for each dataset at `./model/`

#### Visualization Notebooks

We provide visualizations of the results for grouping operations in the `./model/` directory, which include `ETTm1_grouping_reproduce.ipynb`, `mimic27_grouping_reproduce.ipynb`, and `taskonomy_grouping_reproduce.ipynb` notebooks corresponding to the three datasets we used. For each dataset, we provide log files for both training and prediction processes, and grouping is performed using the predicted and sampled values during these processes. The visualized grouping performance results can be seen in these three notebooks. In addition, for the taskonomy dataset, we used an [exhaustive algorithm](https://github.com/tstandley/taskgrouping) for grouping, while for the other two datasets with larger combinatorial spaces, we adopted a strategy to reduce computational complexity for easier grouping operations. The algorithm is introduced in the appendix of the paper and implemented in `search.py`. The provided scripts and notebooks have been tested locally.

### Citation

If you find our work interesting, you can cite the paper as

```text
@inproceedings{
song2022mtgnet,
title={{Efficient and Effective Multi-task Grouping via Meta Learning on Task Combinations},
author={Xiaozhuang Song and Shun Zheng and Wei Cao and James Jianqiao Yu and Jiang Bian},
booktitle={Advances in Neural Information Processing Systems},
year={2022},
}
```