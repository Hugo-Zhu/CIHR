import json
import torch
import pickle
import numpy as np
import pandas as pd
from alegant import logger


with open("dataset/weibo_data.json") as file:
    weibo_data = json.load(file)
with open("dataset/all_user_data.json") as file:
    user_data = json.load(file)

user_weibo_map = {}

for weibo in weibo_data:
    user_id = weibo["user_id"]
    if user_id in user_weibo_map:
        user_weibo_map[user_id].append(weibo)
    else:
        user_weibo_map[user_id] = [weibo]

user_profile_map = {}

for user in user_data:
    user_id = user["user_id"]
    user_profile_map[user_id] = user

logger.info(f"len(user_weibo_map): {user_weibo_map}, len(user_profile_map): {user_profile_map}")

with open("dataset_maps/user_weibo_map.json", "w", encoding='utf-8') as file:
    json.dump(user_weibo_map, file, ensure_ascii=False, indent=4)

with open("dataset_maps/user_profile_map.json", "w", encoding='utf-8') as file:
    json.dump(user_profile_map, file, ensure_ascii=False, indent=4)


profile_feature = np.load('features_bert/user_feature.npy')
profile_feature2 = np.load('features_bert/user_feature2.npy')

with open("dataset/all_user_data.json") as file:
    user_data = json.load(file)

user_feature_map = {}
user_feature_map2 = {}

for i, user in enumerate(user_data):
    user_id = user["user_id"]
    user_feature_map[user_id] = profile_feature[i]
    user_feature_map2[user_id] = profile_feature2[i]

with open("dataset_maps/user_feature_map.pkl", "wb") as file:
    pickle.dump(user_feature_map, file)

with open("dataset_maps/user_feature_map2.pkl", "wb") as file:
    pickle.dump(user_feature_map2, file)