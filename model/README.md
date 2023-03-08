usage example

```
python MTG.py --dataset mimic27 --gpu_id 0 --layer_num 4 --seed 72 --strategy active --num_hidden 64
```

For how to perform grouping operations, please refer to the notebooks we provided. BTW, due to the excessive redundant plotting statements in the notebooks :( (it might get refactored in a future version), you may only need to call the grouping function in `search.py` according to the corresponding input format for use.