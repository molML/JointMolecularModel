
import torch
from torch.utils.data import RandomSampler
from torch.utils.data.dataloader import DataLoader
from collections import defaultdict
import time


class Trainer:

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)
        self.device = config.device

        self.model = self.model.to(self.device)
        print("running on device", self.device)

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

    def run(self, sampling: bool = False, shuffle: bool = True):
        model, config = self.model, self.config

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)) if sampling else None,
            shuffle=False if sampling else shuffle,
            pin_memory=True,
            batch_size=config.batch_size,
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
            batch.to(self.device)
            x = batch.squeeze()
            # print(x.shape)

            # forward the model
            x_hat, z, sample_likelihood, self.loss = self.model(x.float())

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
