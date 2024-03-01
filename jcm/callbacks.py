

import os
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import RandomSampler
from jcm.utils import to_binary, ClassificationMetrics, logits_to_pred, predict_and_eval_mlp


# TODO outfile names

def vae_batch_end_callback(trainer):

    config = trainer.config

    if trainer.iter_num % config.batch_end_callback_every == 0 and trainer.iter_num > 0:
        balanced_accuracies = []
        losses = []

        if config.out_path is not None:
            ckpt_path = os.path.join(config.out_path, f"pretrained_vae_{trainer.iter_num}.pt")
            torch.save(trainer.model.state_dict(), ckpt_path)

        val_loader = DataLoader(trainer.val_dataset,
                                sampler=RandomSampler(trainer.val_dataset, replacement=True,
                                                      num_samples=trainer.config.val_molecules_to_sample),
                                shuffle=False, pin_memory=True, batch_size=trainer.config.batch_size)

        trainer.model.eval()
        for batch in tqdm(val_loader):
            batch.to(config.device)
            x = batch.squeeze()

            x_hat, z, sample_likelihood, loss = trainer.model(x.float())
            losses.append(loss.item())

            x_hat_bin = to_binary(x_hat)
            batch_baccs = [ClassificationMetrics(x[i], x_hat_bin[i]).balanced_accuracy() for i in range(len(x))]
            balanced_accuracies.extend(batch_baccs)

        trainer.model.train()

        mean_val_loss = sum(losses) / len(losses)
        mean_balanced_accuracies = sum(balanced_accuracies) / len(balanced_accuracies)

        trainer.history['iter_num'].append(trainer.iter_num)
        trainer.history['train_loss'].append(trainer.loss.item())
        trainer.history['val_loss'].append(mean_val_loss)
        trainer.history['val_ba'].append(mean_balanced_accuracies)

        print(f"Iter: {trainer.iter_num}, train loss: {round(trainer.loss.item(), 4)}, "
              f"val loss: {round(mean_val_loss, 4)}, balanced accuracy: {round(mean_balanced_accuracies, 4)}")

        if trainer.config.out_path is not None:
            history_path = os.path.join(config.out_path, f"training_history.csv")
            trainer.get_history(history_path)


def mlp_batch_end_callback(trainer):

    config = trainer.config

    if trainer.iter_num % config.batch_end_callback_every == 0 and trainer.iter_num > 0:
        y_hats = []
        y_trues = []
        losses = []

        if config.out_path is not None:
            ckpt_path = os.path.join(config.out_path, f"mlp_{trainer.iter_num}.pt")
            torch.save(trainer.model.state_dict(), ckpt_path)

        val_loader = DataLoader(trainer.val_dataset, shuffle=False, pin_memory=True, batch_size=config.batch_size)

        trainer.model.eval()
        for x, y in val_loader:
            x.to(config.device)
            y.to(config.device)
            x = x.squeeze().float()
            y = y.squeeze()

            y_hat, loss = trainer.model(x, y)

            losses.append(loss.item())
            y_hats.append(y_hat)
            y_trues.append(y)

        trainer.model.train()

        # merge the predictions over batches
        mean_val_loss = sum(losses) / len(losses)
        y_hats = torch.cat(y_hats, 0)
        y_trues = torch.cat(y_trues, 0)

        # compute balanced accuracy
        preds, uncertainty = logits_to_pred(y_hats, return_binary=True, return_uncertainty=True)
        ba = ClassificationMetrics(y=y_trues, y_hat=preds).BA

        # write to file
        trainer.history['iter_num'].append(trainer.iter_num)
        trainer.history['train_loss'].append(trainer.loss.item())
        trainer.history['val_loss'].append(mean_val_loss)
        trainer.history['val_ba'].append(ba)

        print(f"Iter: {trainer.iter_num}, train loss: {round(trainer.loss.item(), 4)}, "
              f"val loss: {round(mean_val_loss, 4)}, balanced accuracy: {round(ba, 4)}")

        if config.out_path is not None:
            history_path = os.path.join(config.out_path, f"training_history.csv")
            trainer.get_history(history_path)


def jvae_batch_end_callback(trainer):

    config = trainer.config

    if trainer.iter_num % config.batch_end_callback_every == 0 and trainer.iter_num > 0:

        losses = []
        val_loader = DataLoader(trainer.val_dataset,
                                sampler=RandomSampler(trainer.val_dataset, replacement=True,
                                                      num_samples=trainer.config.val_molecules_to_sample),
                                shuffle=False, pin_memory=True, batch_size=trainer.config.batch_size)

        trainer.model.eval()
        for x, y in val_loader:

            x.to(trainer.config.device)
            y.to(trainer.config.device)
            x = x.squeeze().float()
            y = y.squeeze()

            y_hat, z, sample_likelihood, loss = trainer.model(x, y)
            losses.append(loss.item())

        trainer.model.train()
        mean_val_loss = sum(losses) / len(losses)
        print(trainer.iter_num, trainer.loss.item(), mean_val_loss)
