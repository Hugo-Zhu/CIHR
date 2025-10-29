import json

with open("dataset/weibo_data.json") as file:
    weibo_data = json.load(file)

with open("dataset/weibo_data_mask.json") as f:
    masks = json.load(f)

mask_train = masks['train']
mask_dev = masks['dev']
mask_test = masks['test']

train_data = [weibo_data[i] for i in range(len(weibo_data)) if mask_train[i]]
dev_data = [weibo_data[i] for i in range(len(weibo_data)) if mask_dev[i]]
test_data = [weibo_data[i] for i in range(len(weibo_data)) if mask_test[i]]

train_file = 'dataset_new/train_data.json'
dev_file = 'dataset_new/dev_data.json'
test_file = 'dataset_new/test_data.json'

with open(train_file, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open(dev_file, 'w', encoding='utf-8') as f:
    json.dump(dev_data, f, ensure_ascii=False, indent=4)

with open(test_file, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)
