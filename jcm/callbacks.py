
import os
from os.path import join as ospj
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import RandomSampler
from jcm.utils import ClassificationMetrics, logits_to_pred, single_batchitem_fix, reconstruction_metrics, \
    logits_to_smiles
from eval.design_eval import strip_smiles, draw_mol_comparison, smiles_validity, reconstruction_edit_distance
from dataprep.descriptors import encoding_to_smiles, probs_to_encoding


def should_perform_callback(interval, i):

    if interval is not None:
        if i % interval == 0 and i > 0:
            return True
    return False


def denovo_lstm_callback(trainer):
    config = trainer.config
    i = trainer.iter_num

    # Check if we want to perform a callback
    if should_perform_callback(config.batch_end_callback_every, i):

        # Save model checkpoint
        if config.out_path is not None:
            trainer.model.save(os.path.join(config.out_path, f"denovo_lstm_{trainer.iter_num}.pt"))

        # Predict from the validation set
        all_probs, all_embeddings, all_sample_losses = trainer.model.predict(trainer.val_dataset, sample=True)
        designs = encoding_to_smiles(probs_to_encoding(all_probs))

        # Get the losses
        val_loss = torch.mean(all_sample_losses)
        train_loss = trainer.loss.item()

        # Update the training history and save if a path is given in the config
        trainer.append_history(iter_num=trainer.iter_num, train_loss=train_loss, val_loss=val_loss)
        if trainer.config.out_path is not None:
            trainer.get_history(os.path.join(config.out_path, f"training_history.csv"))

        print(f"Iter: {i}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, example: {designs[0]}")


# @torch.no_grad()
# def vae_batch_end_callback(trainer):
#
#     config = trainer.config
#     if config.batch_end_callback_every is not None:
#         if trainer.iter_num % config.batch_end_callback_every == 0 and trainer.iter_num > 0:
#             losses = []
#
#             if config.out_path is not None:
#                 ckpt_path = os.path.join(config.out_path, f"pretrained_vae_{trainer.iter_num}.pt")
#                 torch.save(trainer.model.state_dict(), ckpt_path)
#
#             val_loader = DataLoader(trainer.val_dataset,
#                                     sampler=RandomSampler(trainer.val_dataset, replacement=True,
#                                                           num_samples=trainer.config.val_molecules_to_sample),
#                                     shuffle=False, pin_memory=True, batch_size=trainer.config.batch_size,
#                                     collate_fn=single_batchitem_fix)
#
#             xs, xhats = [], []
#
#             trainer.model.eval()
#             for batch in tqdm(val_loader):
#                 batch.to(config.device)
#                 x = batch
#
#                 x_hat, z, sample_likelihood, loss = trainer.model(x)
#
#                 losses.append(loss.item())
#                 xs.append(x)
#                 xhats.append(x_hat)
#
#             trainer.model.train()
#
#             mean_val_loss = sum(losses) / len(losses)
#             mean_metrics = reconstruction_metrics(torch.cat(xhats), torch.cat(xs))
#
#             trainer.history['iter_num'].append(trainer.iter_num)
#             trainer.history['train_loss'].append(trainer.loss.item())
#             trainer.history['val_loss'].append(mean_val_loss)
#
#             for k, v in mean_metrics.items():
#                 trainer.history[f"val_{k}"].append(v)
#
#             print(f"Iter: {trainer.iter_num}, train loss: {round(trainer.loss.item(), 4)}, "
#                   f"val loss: {round(mean_val_loss, 4)}, balanced accuracy: {round(mean_metrics['BA'], 4)}, "
#                   f"100% reconstruction: {round(mean_metrics['recons_100'], 4)}, "
#                   f"99% reconstruction {round(mean_metrics['recons_99'], 4)}, recall: {round(mean_metrics['TPR'], 4)},"
#                   f" precision: {round(mean_metrics['PPV'], 4)}")
#
#             if trainer.config.out_path is not None:
#                 history_path = os.path.join(config.out_path, f"training_history.csv")
#                 trainer.get_history(history_path)
#
#
# @torch.no_grad()
# def mlp_batch_end_callback(trainer):
#
#     config = trainer.config
#     if config.batch_end_callback_every is not None:
#         if trainer.iter_num % config.batch_end_callback_every == 0 and trainer.iter_num > 0:
#             y_hats = []
#             y_trues = []
#             losses = []
#
#             if config.out_path is not None:
#                 ckpt_path = os.path.join(config.out_path, f"mlp_{trainer.iter_num}.pt")
#                 torch.save(trainer.model.state_dict(), ckpt_path)
#
#             val_loader = DataLoader(trainer.val_dataset, shuffle=False, pin_memory=True, batch_size=config.batch_size,
#                                     collate_fn=single_batchitem_fix)
#
#             trainer.model.eval()
#             for x, y in val_loader:
#                 x.to(config.device)
#                 y.to(config.device)
#
#                 y_hat, loss = trainer.model(x, y)
#
#                 losses.append(loss.item())
#                 y_hats.append(y_hat)
#                 y_trues.append(y)
#
#             trainer.model.train()
#
#             # merge the predictions over batches
#             mean_val_loss = sum(losses) / len(losses)
#             y_hats = torch.cat(y_hats, 0)
#             y_trues = torch.cat(y_trues, 0)
#
#             # compute balanced accuracy
#             preds, uncertainty = logits_to_pred(y_hats, return_binary=True, return_uncertainty=True)
#             ba = ClassificationMetrics(y=y_trues, y_hat=preds).BA
#
#             # write to file
#             trainer.history['iter_num'].append(trainer.iter_num)
#             trainer.history['train_loss'].append(trainer.loss.item())
#             trainer.history['val_loss'].append(mean_val_loss)
#             trainer.history['val_ba'].append(ba)
#
#             print(f"Iter: {trainer.iter_num}, train loss: {round(trainer.loss.item(), 4)}, "
#                   f"val loss: {round(mean_val_loss, 4)}, balanced accuracy: {round(ba, 4)}")
#
#             if config.out_path is not None:
#                 history_path = os.path.join(config.out_path, f"training_history.csv")
#                 trainer.get_history(history_path)
#
#
# @torch.no_grad()
# def jvae_batch_end_callback(trainer):
#
#     config = trainer.config
#     if config.batch_end_callback_every is not None:
#         if trainer.iter_num % config.batch_end_callback_every == 0 and trainer.iter_num > 0:
#
#             losses = []
#             xs, x_hats = [], []
#             ys, y_hats = [], []
#             val_loader = DataLoader(trainer.val_dataset,
#                                     sampler=RandomSampler(trainer.val_dataset, replacement=True,
#                                                           num_samples=trainer.config.val_molecules_to_sample),
#                                     shuffle=False, pin_memory=True, batch_size=trainer.config.batch_size,
#                                     collate_fn=single_batchitem_fix)
#
#             trainer.model.eval()
#             for x, y in val_loader:
#
#                 x.to(trainer.config.device)
#                 y.to(trainer.config.device)
#
#                 y_logits_N_K_C, x_hat, z, sample_likelihood, loss = trainer.model(x, y)
#
#                 losses.append(loss.item())
#                 xs.append(x)
#                 x_hats.append(x_hat)
#                 ys.append(y)
#                 y_hats.append(y_logits_N_K_C)
#
#             trainer.model.train()
#
#             mean_val_loss = sum(losses) / len(losses)
#
#             preds, uncertainty = logits_to_pred(torch.cat(y_hats), return_binary=True, return_uncertainty=True)
#             ba = ClassificationMetrics(y=torch.cat(ys), y_hat=preds).BA
#
#             mean_metrics = reconstruction_metrics(torch.cat(x_hats), torch.cat(xs))
#
#             print(f"Iter: {trainer.iter_num}, train loss: {round(trainer.loss.item(), 4)}, "
#                   f"val loss: {round(mean_val_loss, 4)}, balanced accuracy y: {round(ba, 4)}, balanced accuracy x: "
#                   f"{round(mean_metrics['BA'], 4)}, 100% reconstruction x: {round(mean_metrics['recons_100'], 4)}, "
#                   f"99% reconstruction x: {round(mean_metrics['recons_99'], 4)}, recall x: "
#                   f"{round(mean_metrics['TPR'], 4)}, precision x: {round(mean_metrics['PPV'], 4)}")
#
#
# @torch.no_grad()
# def lstm_vae_batch_end_callback(trainer):
#
#     config = trainer.config
#     if config.batch_end_callback_every is not None:
#         if trainer.iter_num % config.batch_end_callback_every == 0 and trainer.iter_num > 0:
#             losses = []
#
#             if config.out_path is not None:
#                 history_path = ospj(config.out_path, config.experiment_name, f"training_history.csv")
#                 trainer.get_history(history_path)
#
#             val_loader = DataLoader(trainer.val_dataset,
#                                     sampler=RandomSampler(trainer.val_dataset, replacement=True,
#                                                           num_samples=trainer.config.val_molecules_to_sample),
#                                     shuffle=False, pin_memory=True, batch_size=trainer.config.batch_size,
#                                     collate_fn=single_batchitem_fix)
#
#             xs, xhats = [], []
#
#             trainer.model.eval()
#             for batch in tqdm(val_loader, disable=True):
#                 x = batch.to(config.device)
#
#                 x_hat, z, sample_likelihood, loss = trainer.model(x)
#
#                 losses.append(loss.item())
#                 xs.append(x)
#                 xhats.append(x_hat)
#
#             trainer.model.train()
#
#             xhats = torch.cat(xhats)
#             xs = torch.cat(xs)
#             mean_val_loss = sum(losses) / len(losses)
#
#             raw_designs = logits_to_smiles(xhats)
#             designs = strip_smiles(raw_designs)
#             # print(xs[0])
#             target_smiles = [encoding_to_smiles(row.tolist()) for row in xs]
#
#             validity, valid_smiles = smiles_validity(designs, return_invalids=True)
#             valid_idx = [i for i, smi in enumerate(valid_smiles) if smi is not None]
#
#             edist = [reconstruction_edit_distance(p_smi, t_smi) for p_smi, t_smi in zip(designs, target_smiles)]
#
#             trainer.history['iter_num'].append(trainer.iter_num)
#             trainer.history['train_loss'].append(trainer.loss.item())
#             trainer.history['val_loss'].append(mean_val_loss)
#             trainer.history['validity'].append(validity)
#             trainer.history['edit_distance'].append(np.mean(edist))
#             trainer.history['random_output'].append(raw_designs[0])
#
#             # Update the LR scheduler with the most recent validation loss
#             # trainer.scheduler.step(mean_val_loss)
#
#             print(f"Iter: {trainer.iter_num}, "
#                   f"train loss: {round(trainer.loss.item(), 4)}, "
#                   f"val loss: {round(mean_val_loss, 4)}, "
#                   f"validity: {round(validity, 4)}, "
#                   f"mean edit distance: {round(np.mean(edist), 4)}, "
#                   # f"lr: {trainer.scheduler.get_last_lr()[0]:e}, "
#                   f"lr: {trainer.get_lr():e}, "
#                   f"example SMILES: {raw_designs[0]}")
#
#             if len(valid_idx) > 0 and config.draw_mol:
#                 mol_to_draw = valid_idx[np.random.randint(0, len(valid_idx), 1)[0]]
#                 draw_mol_comparison(valid_smiles[mol_to_draw], target_smiles[mol_to_draw])

