We use different backbones for the multi-task setting under different scenarios. We adapted the code from [Which Tasks to Train Together in Multi-Task Learning](https://github.com/tstandley/taskgrouping), [EHR-MTL-benchmark](https://github.com/mmcdermott/comprehensive_MTL_EHR), and [Autoformer](https://github.com/thuml/Autoformer) for vision tasks, healthcare tasks, and energy tasks respectively.

Our experiments are configured according to the requirements within this code repository for the experimental environment and dataset.

#### Baselines

- For HOA, we can use the pairwise ground-truth gains to derive the task transferring gains.
- For TAG, we provided scripts that make some modifications to the backbone networks for each scenario, making it efficiently collect *task affinities* in this process and derive a final task transferring gains estimate based on HOA's algorithm.