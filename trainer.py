from typing import Callable, Dict, List, Optional, Tuple
from transformers import DataCollatorForSeq2Seq, PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset
import torch
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from IPython.display import clear_output
import matplotlib.pyplot as plt
import inspect

import numpy as np
import os
import gc


class TrainerCallbacksList():
    def __init__(self, on_startup_callbacks: List[Callable] = [],
                 on_eval_end_callback: List[Callable] = [],
                 on_start_train_callbacks: List[Callable] = []):
        self.on_startup_callbacks = on_startup_callbacks
        self.on_eval_end_callback = on_eval_end_callback
        self.on_start_train_callbacks = on_start_train_callbacks

    def run_startup_callbacks(self, trainer_instance):
        for call in self.on_startup_callbacks:
            call(self, trainer_instance)

    def run_eval_end_callbacks(self, trainer_instance, **kwargs):
        for call in self.on_eval_end_callback:
            call(self, trainer_instance, **kwargs)

    def run_start_train_callbacks(self, trainer_instance, **kwargs):
        for call in self.on_start_train_callbacks:
            call(self, trainer_instance, **kwargs)


class Seq2SeqTrainer():
    def __init__(self, train_dataset: Dataset, val_dataset: Dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                 batch_size, num_epoch, grad_acc_step=1, max_grad_norm=1,  eval_batch_size=None,
                 max_eval_batches=float('inf'), eval_generation_kwargs: Dict = {'num_beams': 1, 'max_new_tokens': 30},
                 lr=5e-5, weight_decay=0,
                 optimizer=None, scheduler=None,
                 metrics: List[Tuple[str, Callable]] = [],
                 eval_freq=5000, saving_path='/', model_name='model', sampler=None, fp16=False,
                 callbacks: Optional[TrainerCallbacksList] = None, description=None, freq_online_loss_plot: int = -1):

        self.description = description
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = model.device
        self.num_epoch = num_epoch
        self.metrics = metrics
        self.eval_freq = eval_freq
        self.saving_path = saving_path
        self.model_name = model_name + '.pt'
        self.max_eval_batches = max_eval_batches
        self.eval_generation_kwargs = eval_generation_kwargs
        self.callbacks = callbacks
        self.writer = SummaryWriter(log_dir=self.saving_path)
        self.fp16 = fp16
        self.grad_acc_step = grad_acc_step
        self.max_grad_norm = max_grad_norm
        self.freq_online_loss_plot = freq_online_loss_plot

        if eval_batch_size is None:
            self.eval_batch_size = batch_size
        else:
            self.eval_batch_size = eval_batch_size

        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        if optimizer is not None:
            self.optimizer = optimizer(
                optimizer_grouped_parameters, lr=lr,)
        else:
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters, lr=lr,)

        self.scheduler = scheduler
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model)

        if sampler is None:
            self.sampler = RandomSampler(self.train_dataset)
        else:
            self.sampler = sampler

        if fp16:
            from apex import amp
            opt_level = 'O1'
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=opt_level)

        configs = f"num_epoch: {self.num_epoch}, batch_size: {self.batch_size},  \n" +\
            f"optimizer: {self.optimizer}  \n" +\
            f"scheduler: {self.scheduler}  \n" +\
            f"grad_acc_step: {grad_acc_step}, max_grad_norm: {max_grad_norm}  \n" +\
            f"generation eval args: {str(eval_generation_kwargs)}  \n" +\
            f"fp16: {fp16}  \n" +\
            f"description: {description}"

        self.writer.add_text("configs", configs, 0)

    def data2device(self, data: Dict):
        return {k: v.to(self.device) for k, v in data.items()}

    def evaluation(self, val_dataloader: DataLoader, max_eval_batches = None, generate_func: Optional[Callable] = None, **eval_generation_kwargs):
        val_loss = 0
        eval_outputs = {"preds": [], "targets": []}
        if max_eval_batches is None:
            max_eval_batches = len(val_dataloader)
            
        for eval_step, val_batch in enumerate(val_dataloader):
            if eval_step > max_eval_batches:
                break

            val_batch = self.data2device(val_batch)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**val_batch)
                val_loss += outputs.loss.item()

                if not generate_func:
                    generate_func = self.model.generate

                outputs = generate_func(input_ids=val_batch["input_ids"],
                                            attention_mask=val_batch["attention_mask"],
                                            **eval_generation_kwargs)

                eval_outputs['preds'].extend(self.tokenizer.batch_decode(
                    outputs, skip_special_tokens=True))

                if 'targets' not in self.val_dataset.column_names:
                    eval_outputs['targets'].extend([[_] for _ in self.tokenizer.batch_decode(
                        val_batch['decoder_input_ids'], skip_special_tokens=True)])

        if not eval_outputs['targets'] and 'targets' in self.val_dataset.column_names:
            eval_outputs['targets'] = self.val_dataset['targets'][:len(eval_outputs['preds'])]

        val_loss = val_loss/max_eval_batches
        return val_loss, eval_outputs

    def train(self):
        if self.fp16:
            from apex import amp

        best_val_loss = float('inf')

        forward_args = inspect.signature(self.model.forward).parameters.keys()

        remove_columns = [c for c in self.train_dataset.column_names if c not in forward_args]
        train_dataloader = DataLoader(self.train_dataset.remove_columns(remove_columns), batch_size=self.batch_size,
                                      collate_fn=self.data_collator,
                                      sampler=self.sampler)

        
        remove_columns = [c for c in self.val_dataset.column_names if c not in forward_args]
        val_dataloader = DataLoader(
            self.val_dataset.remove_columns(remove_columns), batch_size=self.eval_batch_size, collate_fn=self.data_collator, shuffle=False)

        if self.max_eval_batches is None:
            self.max_eval_batches = len(val_dataloader)

        if self.callbacks:
            self.callbacks.run_startup_callbacks(self)

        for epoch in range(self.num_epoch):
            self.epoch = epoch

            train_loss_set = []
            train_loss = 0

            val_loss_set = []
            x_val_set = []

            for step, batch in enumerate(train_dataloader):
                try:
                    # Training
                    self.step = step
                    self.model.train()
                    batch = self.data2device(batch)

                    if self.callbacks:
                        self.callbacks.run_start_train_callbacks(self)

                    outputs = self.model(**batch)
                    loss = outputs.loss

                    if self.fp16:
                        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    if (step + 1) % self.grad_acc_step == 0:

                        if self.fp16:
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer), self.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.max_grad_norm)

                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    train_loss_set.append(loss.item())
                    train_loss += loss.item()

                    if loss.item() is np.NaN:
                        print('Loss is NaN')

                except RuntimeError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue

                # Evaluation
                if step != 0 and (step % self.eval_freq == 0 or step == len(train_dataloader)-1):

                    val_loss, eval_outputs = self.evaluation(
                        val_dataloader, self.max_eval_batches, **self.eval_generation_kwargs)
                    val_loss_set.append(val_loss)
                    x_val_set.append(step)

                    # log metrics and loss to tensorboard
                    for name, func in self.metrics:
                        value = func(**eval_outputs)
                        self.writer.add_scalar(
                            name, value, step + len(train_dataloader)*epoch)

                    self.writer.add_scalar("Loss/val", val_loss,
                                           step + len(train_dataloader)*epoch)
                    self.writer.add_scalar(
                        "Loss/train", train_loss/(step + 1), step + len(train_dataloader)*epoch)

                    # savemodel
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.model.save_pretrained(self.saving_path)
                        self.tokenizer.save_pretrained(self.saving_path)

                    if self.callbacks:
                        self.callbacks.run_eval_end_callbacks(
                            self, **eval_outputs)

                # plot online loss
                if self.freq_online_loss_plot > 0 and step % self.freq_online_loss_plot == 0:
                    clear_output(True)
                    fig, ax = plt.subplots(figsize=(15, 8))
                    plt.grid()
                    plt.plot(train_loss_set)
                    plt.plot(x_val_set, val_loss_set)
                    plt.title("Training/Val loss")
                    plt.xlabel("Batch")
                    plt.ylabel("Loss")
                    plt.show()

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

        print("Train Loss: {0:.5f}".format(train_loss / len(train_dataloader)))
        print("Validation Loss: {0:.5f}".format(val_loss))
        self.writer.close()
