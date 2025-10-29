import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from transformers import get_linear_schedule_with_warmup
from alegant import logger, seed_everything
from alegant.trainer import TrainingArguments, Trainer, DataModule, autocast, GradScaler
from src.dataset import DataModule
from src.utils import WarmupLR
from sklearn.metrics import classification_report


class BertTrainer(Trainer):
    def __init__(self,
        args: TrainingArguments, 
        model: nn.Module, 
        data_module: DataModule):       
        super(BertTrainer, self).__init__(args, model, data_module)

        self.criterion = nn.CrossEntropyLoss()


    def training_step(self, batch, batch_idx):
        with autocast(enabled=self.args.amp):
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            logits = self.model(**batch)
            loss = self.criterion(logits, batch["label"])
        return {"loss": loss}


    def validation_step(self, batch, batch_idx):
        with autocast(enabled=self.args.amp):
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            logits = self.model(**batch)
            loss = self.criterion(logits, batch["label"])
        return {"loss": loss, "logits": logits, "labels": batch["label"]}


    def validation_epoch_end(self, outputs):
        logits_all_sample = torch.cat([item["logits"].detach().cpu() for item in outputs])
        preds_all_sample = torch.argmax(F.softmax(logits_all_sample, dim=-1), dim=-1)
        labels_all_sample = torch.cat([item["labels"].detach().cpu() for item in outputs])
        metrics = classification_report(labels_all_sample, preds_all_sample, output_dict=True)
        avg_f1 = metrics['macro avg']['f1-score']

        self.logger.add_scalar("avg_f1", avg_f1, self.num_eval)
        
        # Update best metrix
        if avg_f1 > self.best_f1:
            self.best_f1 = avg_f1
            if self.args.do_save:
                self.save_checkpoint()
        logger.success(f"\n avg_f1: {avg_f1}, best_f1: {self.best_f1}")


    def test_epoch_end(self, outputs):
        logits_all_sample = torch.cat([item["logits"].detach().cpu() for item in outputs])
        preds_all_sample = torch.argmax(F.softmax(logits_all_sample, dim=-1), dim=-1)
        labels_all_sample = torch.cat([item["labels"].detach().cpu() for item in outputs])
        metrics = classification_report(labels_all_sample, preds_all_sample, output_dict=True)

        logger.info(metrics)


    def configure_optimizers(self, num_training_steps):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        extra_params = ['linear_beta', 'linear_gamma', 'Linear_mu', 'Linear_sigma', 'attribute_embeddings']

        # Parameter groups: 1. parms of PLM, 2.params of other components
        params_plm = []
        no_decay_params_plm = []
        params = []
        no_decay_params = []
        params_extra = []
        no_decay_params_extra = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad == False:
                continue
            elif any(e in name for e in extra_params):
                if any(nd in name for nd in no_decay):
                    no_decay_params_extra.append(param)
                else:
                    params_extra.append(param)
            elif "bert" in name and not any(e in name for e in extra_params):
                if any(nd in name for nd in no_decay):
                    no_decay_params_plm.append(param)
                else:
                    params_plm.append(param)
            else:
                if any(nd in name for nd in no_decay):
                    no_decay_params.append(param)
                else:
                    params.append(param)

        # Grouped parameters
        optimizer_grouped_parameters = [
            {
                "params": params_extra,
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate
            },
            {
                "params": no_decay_params_extra,
                "weight_decay": 0.0,
                "lr": self.args.learning_rate
            },
            {
                "params": params_plm,
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate_plm
            },
            {   
                "params": no_decay_params_plm,
                "weight_decay": 0.0,
                "lr": self.args.learning_rate_plm
            },
            {
                "params": params,
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate
            },
            {
                "params": no_decay_params, 
                "weight_decay": 0.0,
                "lr": self.args.learning_rate
            }
        ]

        # Define the optimizer
        optimizer_ = Adam(optimizer_grouped_parameters[0:4])
        optimizer2 = Adam(optimizer_grouped_parameters[4:])

        # Define the scheduler
        scheduler_ = WarmupLR(optimizer=optimizer_, warmup_steps=int(num_training_steps*0.2), total_steps=num_training_steps, initial_lr=self.args.learning_rate, final_lr=self.args.learning_rate_plm)

        return [optimizer_, optimizer2], [scheduler_]


if __name__ == "__main__":
    seed_everything(0)
    import sys
    from dataset_multi_view.load_data import full_dataset
    logger.remove()
    logger.add(sys.stdout, format='File "<cyan>{file.path}</cyan>", line <cyan>{line}</cyan>\n  {message}', level='INFO')

    from src.modeling import Model
    from src.dataset import DataModuleConfig, DataModule
    data_module_config = DataModuleConfig(
        train_batch_size=2,
        train_limit_batches=1.0,
        val_batch_size=32,
        val_limit_batches=1.0,
        test_batch_size=32,
        test_limit_batches=1.0
    )
    data_module = DataModule(data_module_config, full_dataset)
    model = Model(777)
    training_args = TrainingArguments(
        epochs=50,
        device="cuda",
        gpus=[0, 1],
        accumulation_steps=16,
        log_dir="./tensorboard/attribute_history_multi-view"
    )
    trainer = BertTrainer(
        training_args, 
        model, 
        data_module
    )
    trainer.fit()