import torch
from torch import nn
from torch.nn import functional as F
from modeling_bert import CrossAttention
from modeling_bert import BertModel as MyBertModel
from alegant import logger


class WeightedFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(2 * d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        """
        x1/x2 --> output: [batch_size, num_nodes, d_model]
        """
        alpha = self.sigmoid(self.linear(torch.cat((x1, x2), dim=-1)))
        output = alpha * x1 + (1 - alpha) * x2

        return output

class Model(nn.Module):
    def __init__(self, in_features_attribute, dropout=0.1, *args, **kwargs):
        super(Model, self).__init__()
        self.bert = MyBertModel.from_pretrained('hfl/chinese-bert-wwm-ext')
        self.attribute_embeddings = nn.Linear(in_features_attribute, self.bert.config.hidden_size)
        nn.init.zeros_(self.attribute_embeddings.weight)
        nn.init.zeros_(self.attribute_embeddings.bias)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)
        self.cross_attn = CrossAttention(
            input_size=self.bert.config.hidden_size,
            hidden_size=self.bert.config.hidden_size,
            n_attn_heads=4,
            cross_attn_type=5,
            use_forget_gate=True,
        )
        self.gate_fuse = WeightedFusion(self.bert.config.hidden_size)

    def forward(self, attributes,
                input_ids, attention_mask, history, history_mask, **kwargs):
        """
        attributes:     [batch_size, in_features_attribute                  ]
        input_ids:      [batch_size,    num_seq_text,       seq_len_text    ]
        attention_mask: [batch_size,    num_seq_text,       seq_len_text    ]
        history:        [batch_size,    num_seq_history,    seq_len_history ]
        history_mask:   [batch_size,    num_seq_history,    seq_len_history ]
        """
        # check shapes
        assert len(attributes.shape) == 2
        assert len(input_ids.shape) == 3
        assert len(history.shape) == 3
        assert input_ids.shape == attention_mask.shape
        assert history.shape == history_mask.shape
        assert input_ids.shape[0] == history.shape[0]
        assert input_ids.shape[0] == attributes.shape[0]

        message = f"attributes.shape: {attributes.shape}\n"
        message += f"input_ids.shape: {input_ids.shape}\n"
        message += f"attention_mask.shape: {attention_mask.shape}\n"
        message += f"history.shape: {history.shape}\n"
        message += f"history_mask.shape: {history_mask.shape}\n"
        logger.debug(message)

        attribute_embeddings = self.attribute_embeddings(attributes).unsqueeze(1)
        logger.debug(f"attribute_embeddings.shape: {attribute_embeddings.shape}")

        batch_size, num_seq_text, seq_len_text = input_ids.shape
        input_ids = input_ids.view(batch_size * num_seq_text, seq_len_text)
        attention_mask = attention_mask.view(batch_size * num_seq_text, seq_len_text)
        outputs = self.bert(attribute_embeddings=attribute_embeddings, input_ids=input_ids, attention_mask=attention_mask)
        # pooled_output = outputs.pooler_output
        pooled_output = outputs.last_hidden_state[:,0,:]
        pooled_output = pooled_output.view(batch_size, num_seq_text, -1)
        logger.debug(f"pooled_output.shape: {pooled_output.shape}")

        batch_size, num_seq_history, seq_len_history = history.shape
        history = history.view(batch_size * num_seq_history, seq_len_history)
        history_mask = history_mask.view(batch_size * num_seq_history, seq_len_history)
        history_outputs = self.bert(attribute_embeddings=attribute_embeddings, input_ids=history, attention_mask=history_mask)
        history_pooled_output = history_outputs.last_hidden_state[:,0,:]
        history_pooled_output = history_pooled_output.view(batch_size, num_seq_history, -1)
        logger.debug(f"history_pooled_output.shape: {history_pooled_output.shape}")
        
        pooled_output = self.dropout(pooled_output)
        fused_pooled_output = self.cross_attn(history_pooled_output, pooled_output)
        logger.debug(f"Fused pooled_output.shape: {pooled_output.shape}")
        fused_pooled_output = fused_pooled_output.mean(dim=1)   # mean of all texts
        # pooled_output = pooled_output[:, 0, :]
        
        pooled_output = self.gate_fuse(pooled_output.mean(dim=1), fused_pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        logger.debug(f"logits.shape: {logits.shape}")

        return logits
