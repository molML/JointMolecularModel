
from collections import defaultdict
import time
import pandas as pd
import torch
from torch.utils.data import RandomSampler
from torch.utils.data.dataloader import DataLoader
from jcm.model import VAE, JVAE, Ensemble
from jcm.callbacks import vae_batch_end_callback, mlp_batch_end_callback, jvae_batch_end_callback
from jcm.utils import single_batchitem_fix


class Trainer:

    def __init__(self, config, model, train_dataset, val_dataset=None):
        self.config = config
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.callbacks = defaultdict(list)
        self.device = config.device

        self.model = self.model.to(self.device)
        # print("running on device", self.device)

        self.history = {'iter_num': [], 'train_loss': [], 'val_loss': [], 'val_ba': []}

        # variables for logging
        self.epoch_num = 0
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def get_history(self, out_file: str = None) -> pd.DataFrame:
        """ Get/write training history

        :param out_file: Path of the outputfile (.csv)
        :return: training history
        """
        hist = pd.DataFrame(self.history)

        if out_file is not None:
            hist.to_csv(out_file)
        else:
            return hist

    def run(self, sampling: bool = False, shuffle: bool = True):
        model, config = self.model, self.config

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)) if sampling else None,
            shuffle=False if sampling else shuffle,
            pin_memory=True,
            batch_size=config.batch_size,
            collate_fn=single_batchitem_fix
        )

        model.train()
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
                x = batch[0]
                y = batch[1]
                x.to(self.device)
                y.to(self.device)

            else:
                batch.to(self.device)
                y = None
                x = batch

            # forward the model. The model should always output the loss as the last output here (e.g. (y_hat, loss))
            self.loss = self.model(x, y)[-1]

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num > config.max_iters:
                break


def train_vae(config, train_dataset, val_dataset=None, pre_trained_path: str = None):

    model = VAE(**config.hyperparameters)

    if pre_trained_path is not None:
        model.load_state_dict(torch.load(pre_trained_path))

    T = Trainer(config, model, train_dataset, val_dataset)
    if val_dataset is not None:
        T.set_callback('on_batch_end', vae_batch_end_callback)
    T.run()

    return model, T


def train_mlp(config, train_dataset, val_dataset=None, pre_trained_path: str = None):

    model = Ensemble(**config.hyperparameters)

    if pre_trained_path is not None:
        model.load_state_dict(torch.load(pre_trained_path))

    T = Trainer(config, model, train_dataset, val_dataset)
    if val_dataset is not None:
        T.set_callback('on_batch_end', mlp_batch_end_callback)
    T.run()

    return model, T


def train_jvae(config, train_dataset, val_dataset=None, pre_trained_path_vae: str = None, pre_trained_path_mlp: str = None,
               freeze_vae: bool = False, freeze_mlp: bool = False):

    model = JVAE(**config.hyperparameters)

    if pre_trained_path_vae is not None:
        model.vae.load_state_dict(torch.load(pre_trained_path_vae))

    if pre_trained_path_mlp is not None:
        model.prediction_head.load_state_dict(torch.load(pre_trained_path_mlp))

    if freeze_vae:
        for p in model.vae.parameters():
            p.requires_grad = False

    if freeze_mlp:
        for p in model.prediction_head.parameters():
            p.requires_grad = False

    T = Trainer(config, model, train_dataset, val_dataset)
    if val_dataset is not None:
        T.set_callback('on_batch_end', jvae_batch_end_callback)
    T.run()

    return model, T
