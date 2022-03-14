import pandas as pd

list_eval_partition = pd.read_csv("/data/wangyu/celebA/data/list_eval_partition.csv").rename(columns={"partition":"split"})
list_attr_celeba = pd.read_csv("/data/wangyu/celebA/data/list_attr_celeba.csv")

result = pd.merge(list_eval_partition, list_attr_celeba, on='image_id')
result.to_csv("/data/wangyu/celebA/data/metadata.csv")