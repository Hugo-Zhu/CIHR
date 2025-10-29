import json
import torch
import pickle
import datasets
import numpy as np
import pandas as pd
from alegant import logger


with open("dataset_maps/user_profile_map.json") as file:
    user_profile_map = json.load(file)

with open("dataset_maps/user_weibo_map.json") as file:
    user_weibo_map = json.load(file)

with open("dataset/all_user_data.json") as file:
    user_data = json.load(file)

profile_feature = np.load('features_bert/user_feature.npy')
profile_feature2 = np.load('features_bert/user_feature2.npy')

user_feature_map = {}
user_feature_map2 = {}

for i, user in enumerate(user_data):
    user_id = user["user_id"]
    user_feature_map[user_id] = profile_feature[i]
    user_feature_map2[user_id] = profile_feature2[i]


data_files = {
    "train": "dataset_multi_view/train_data.json",
    "validation": "dataset_multi_view/dev_data.json",
    "test": "dataset_multi_view/test_data.json"
}

full_dataset = datasets.load_dataset("json", data_files=data_files)

full_dataset = full_dataset.map(lambda x: {"user_profile": user_profile_map[str(x["user_id"])]})
full_dataset = full_dataset.map(lambda example: {'history': [x['text'] for x in user_weibo_map[str(example['user_id'])] if x['weibo_id'] != example['weibo_id']]})
full_dataset = full_dataset.map(lambda example: {'user_profile_feature': torch.cat([torch.from_numpy(user_feature_map[example['user_id']]),  torch.from_numpy(user_feature_map2[example['user_id']])])})

logger.debug(full_dataset)