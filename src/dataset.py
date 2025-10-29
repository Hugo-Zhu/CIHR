import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from src.modeling import Model
from alegant import logger
from dataclasses import dataclass, field


tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
def tokenize_texts(example, max_length=512, pad_value=0):
    inputs = tokenizer([example["text"]]*len(example["analysis"]), example["analysis"], truncation=True, padding='max_length', max_length=max_length)
    
    return {
        "input_ids_texts": inputs.input_ids,
        "attention_mask_texts": inputs.attention_mask
    }

def tokenize_history(example, max_num_sequence=20, max_length=128, pad_value=0):
    inputs = tokenizer(example["history"], truncation=True, padding='max_length', max_length=max_length)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    if len(input_ids) > max_num_sequence:
          input_ids = input_ids[:max_num_sequence]
          attention_mask = attention_mask[:max_num_sequence]

    if len(input_ids) < max_num_sequence:
        pad_num = max_num_sequence - len(input_ids)
        input_ids += [[pad_value] * max_length] * pad_num
        attention_mask += [[0] * max_length] * pad_num

    assert (len(input_ids) == max_num_sequence)
    
    return {
        "input_ids_history": input_ids,
        "attention_mask_history": attention_mask
    }

def data_collator(batch):
    attributes = torch.tensor([example['user_profile_feature'] for example in batch])
    input_ids = torch.LongTensor([example['input_ids_texts'] for example in batch])
    attention_mask = torch.LongTensor([example['attention_mask_texts'] for example in batch])
    history = torch.LongTensor([example['input_ids_history'] for example in batch])
    history_mask = torch.LongTensor([example['attention_mask_history'] for example in batch])
    label = torch.tensor([example['humor_level'] for example in batch])
    return {
        'attributes': attributes,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'history': history,
        'history_mask': history_mask,
        'label': label
    }


@dataclass
class DataModuleConfig:
    """
    Data configuration class for defining data-related configuration parameters.
    """
    train_batch_size: int = field(default=32, metadata={"description": "Batch size for training data"})
    train_limit_batches: float = field(default=1.0, metadata={"description": "Batch limit ratio for training data"})
    val_batch_size: int = field(default=32, metadata={"description": "Batch size for validation data"})
    val_limit_batches: float = field(default=1.0, metadata={"description": "Batch limit ratio for validation data"})
    test_batch_size: int = field(default=32, metadata={"description": "Batch size for test data"})
    test_limit_batches: float = field(default=1.0, metadata={"description": "Batch limit ratio for test data"})


class DataModule:
    def __init__(self, config, dataset):
        self.config = config
        tokenized_datasets = dataset.map(tokenize_texts, batched=False)
        tokenized_datasets = tokenized_datasets.map(tokenize_history, batched=False)
        logger.debug(tokenized_datasets)
        self.train_dataset = tokenized_datasets['train']
        self.val_dataset = tokenized_datasets['validation']
        self.test_dataset = tokenized_datasets['test']

    def train_dataloader(self):
        """
        Returns a data loader for training data.

        Returns:
            DataLoader: Data loader for training data.
        """
        full_size = len(self.train_dataset)
        real_size = int(full_size * self.config.train_limit_batches)
        train_dataset, _ = torch.utils.data.random_split(self.train_dataset, [real_size, full_size-real_size])
        train_dataloader = DataLoader(dataset = train_dataset, 
                                shuffle = True,
                                batch_size = self.config.train_batch_size,
                                collate_fn=data_collator)
        return train_dataloader
    
    def val_dataloader(self):
        """
        Returns a data loader for validation data.

        Returns:
            DataLoader: Data loader for validation data.
        """
        full_size = len(self.val_dataset)
        real_size = int(full_size * self.config.val_limit_batches)
        val_dataset, _ = torch.utils.data.random_split(self.val_dataset, [real_size, full_size-real_size])
        val_dataloader = DataLoader(val_dataset, 
                                shuffle = False,
                                batch_size = self.config.val_batch_size,
                                collate_fn=data_collator)
        return val_dataloader

    def test_dataloader(self):
        """
        Returns a data loader for test data.

        Returns:
            DataLoader: Data loader for test data.
        """
        full_size = len(self.test_dataset)
        real_size = int(full_size * self.config.test_limit_batches)
        test_dataset, _ = torch.utils.data.random_split(self.test_dataset, [real_size, full_size-real_size])
        test_dataloader = DataLoader(test_dataset, 
                                shuffle = False,
                                batch_size = self.config.test_batch_size,
                                collate_fn=data_collator)
        return test_dataloader
