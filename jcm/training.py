import os
import time
import warnings
from os.path import join as ospj
from collections import defaultdict
import pandas as pd
import torch
from jcm.config import save_settings
from torch.utils.data import RandomSampler, WeightedRandomSampler
from torch.utils.data.dataloader import DataLoader
from jcm.utils import single_batchitem_fix
from jcm.callbacks import should_perform_callback
import numpy as np


class Trainer:

    def __init__(self, config, model, train_dataset, val_dataset=None, save_models: bool = True):
        self.config = config
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.callbacks = defaultdict(list)
        self.device = config.device
        self.outdir = None
        self.save_models = save_models

        self.model = self.model.to(self.device)
        self.history = defaultdict(list)

        # variables for logging
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

        print(f'Training on {self.device}')

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def append_history(self, **kwargs):
        for k, v in kwargs.items():
            self.history[k].append(v)

    def prep_outdir(self):
        if self.outdir is None:
            if self.config.out_path is not None:
                # make the dir and update the variable if succeeded
                outdir = ospj(self.config.out_path)
                os.makedirs(outdir, exist_ok=True)

                for f in os.listdir(outdir):
                    if f.startswith('checkpoint'):
                        os.remove(ospj(outdir, f))
                    if f == 'training_history.csv':
                        os.remove(ospj(outdir, f))

                # save config file to the outdir
                save_settings(self.config, ospj(outdir, 'experiment_settings.yml'))

                self.outdir = outdir

    def get_history(self, out_file: str = None) -> pd.DataFrame:
        """ Get/write training history

        :param out_file: Path of the outputfile (.csv)
        :return: training history
        """
        hist = pd.DataFrame(self.history)

        if out_file is not None:
            hist.to_csv(out_file, index=False)
        else:
            return hist

    def keep_best_model(self, load_weights: bool = False):
        """ Load the weights of the best model checkpoint (by validation loss) and delete all other checkpoints """

        checkpoints = [f for f in os.listdir(self.outdir) if f.startswith('checkpoint')]

        if len(checkpoints) > 1:

            # find the best checkpoint
            best_checkpoint = f"checkpoint_{self.history['iter_num'][np.argmin(self.history['val_loss'])]}.pt"

            # load best weights
            if load_weights:
                print(f"Loading: {best_checkpoint}")
                self.model.load_weights(ospj(self.outdir, best_checkpoint))

            # delete the other checkpoints
            for ckpt in checkpoints:
                if ckpt != best_checkpoint:
                    os.remove(ospj(self.outdir, ckpt))

        # If there's only one checkpoint, then just load that one
        else:
            if load_weights:
                print(f"Loading: {checkpoints[0]}")
                self.model.load_weights(ospj(self.outdir, checkpoints[0]))

    def run(self, sampling: bool = True, shuffle: bool = True):
        model, config = self.model, self.config

        # create the output dir and clean it up if it already exists
        self.prep_outdir()

        # define the data sampler: random or weighted_random
        sampler = None
        if sampling:
            if config.balance_classes:
                weights = get_balanced_sample_weights(self.train_dataset)
                sampler = WeightedRandomSampler(weights, replacement=True, num_samples=len(self.train_dataset))
            else:
                sampler = RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10))

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=sampler,
            shuffle=False if sampling else shuffle,
            pin_memory=True,
            batch_size=config.batch_size,
            collate_fn=single_batchitem_fix
        )

        # initiate training
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            if len(batch) == 2:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
            else:
                y = None
                x = batch.to(self.device)

            # The model should always output the loss as the last output here (e.g. (y_hat, loss))
            model.train()
            self.loss = model(x, y)[-1]

            if torch.isnan(self.loss):
                warnings.warn('Skipping mini batch due to Nan Loss')
            else:
                self.loss.backward()

                # clip gradients
                if config.grad_norm_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_norm_clip)

                self.optimizer.step()
            self.optimizer.zero_grad()

            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow
            self.iter_num += 1

            # perform validation and update some variables
            self.trigger_callbacks('on_batch_end')

            # Check if we should do some early stopping
            if len(self.callbacks) > 0:
                if not has_improved_in_n(self.history[config.early_stopping_metric],
                                         n=config.early_stopping_patience,
                                         eps=config.early_stopping_eps,
                                         should_go_down=config.early_stopping_should_go_down):

                    print(f"Stopping early at iter {self.iter_num}")
                    break

            # save model
            if self.config.out_path is not None:
                if should_perform_callback(config.batch_end_callback_every, self.iter_num):
                    ckpt_path = ospj(self.outdir, f"checkpoint_{self.iter_num}.pt")
                    model.save_weights(ckpt_path)

                    # delete all models that are not 'the best' to save memory
                    if config.keep_best_only:
                        self.keep_best_model(load_weights=False)

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break

        # load the best weights and get rid of the suboptimal model checkpoints
        if config.keep_best_only and self.config.out_path is not None:
            self.keep_best_model(load_weights=True)

        if not self.save_models:
            checkpoints = [f for f in os.listdir(self.outdir) if f.startswith('checkpoint')]
            for ckpt in checkpoints:
                os.remove(ospj(self.outdir, ckpt))


def has_improved_in_n(metric: list, n: int = 5, should_go_down: bool = True, eps: float = 0) -> bool:
    """ Early stopping check function. Checks if the last n entries of a metric are lower or higher than the lowest/
    highest point before the last n.

    :param metric: list/array of any metric that needs to be monitored
    :param n: patience
    :param should_go_down: True if metric needs to go down (e.g. loss), False if metric should go up (e.g. accuracy)
    :eps minimal difference (default = 0)
    :return: bool
    """

    if len(metric) <= n:
        return True

    before_n = metric[:-n]
    last_n = metric[-n:]

    if should_go_down:
        return (min(last_n) + eps) < min(before_n)
    else:
        return (max(last_n) - eps) > max(before_n)


def get_balanced_sample_weights(dataset) -> list[float]:
    """ Get the sampling probability for each sample inversely proportional to their class ratio so classes will be
    balanced during sampling.

    :param dataset: A torch dataset object
    :return: list of probabilities
    """

    class_weights = [1 - sum((dataset.y == 0) * 1) / len(dataset.y), 1 - sum((dataset.y == 1) * 1) / len(dataset.y)]
    weights = [class_weights[i].item() for i in dataset.y]

    return weights
