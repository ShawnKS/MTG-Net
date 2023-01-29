## MTG-Net

The repository contains the codes for the paper, Efficient and Effective Multi-task Grouping via Meta Learning on Task Combinations, which is accepted by Advances in Neural Information Processing Systems (NeurIPS) 2022.

### Dataset & Backbone Model

#### Taskonomy

We follow the backbone model used in [Which Tasks to Train Together in Multi-Task Learning](https://github.com/tstandley/taskgrouping) (ICML'20)

#### MIMICIII

We follow the backbone model used in [A Comprehensive EHR Timeseries Pre-training Benchmark](https://dl.acm.org/doi/pdf/10.1145/3450439.3451877) (CHIL'21)

#### ETTm1

We follow the backbone model used in [Autoformer](https://github.com/thuml/Autoformer) (NeurIPS21)

### Transferring Gain Collection

We provide scripts for collecting the transferring gain [here](./gain_collection).

### Baselines

#### HOA

The HOA algorithm can be calculated using pairwise task transferring gains, we provide examples in ./gain_collection directory.

#### TAG

We provide each datasets gain collection with a TAG version. You can refer to the readme file under ./gain_collection directory for details.

### Pipeline

1. Collect ground truth transferring gains on datasets.
2. Run MTG-Net script to obtain the predicted gain.
3. Run grouping script to do the final grouping.

(contact for details in all settings)