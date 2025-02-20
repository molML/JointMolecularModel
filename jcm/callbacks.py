
import os
import numpy as np
from cheminformatics.encoding import strip_smiles, probs_to_smiles
from cheminformatics.eval import smiles_validity, reconstruction_edit_distance, plot_molecular_reconstruction
from jcm.utils import logits_to_pred
from sklearn.metrics import balanced_accuracy_score
import warnings
import wandb
import multiprocessing
from multiprocessing import Queue
from cheminformatics.utils import smiles_to_mols

# Ignore specific UserWarning from sklearn
warnings.filterwarnings(action='ignore', category=UserWarning, message="y_pred contains classes not in y_true")
warnings.filterwarnings(action='ignore', category=UserWarning, message="A single label was found in 'y_true' and 'y_pred'")


def execute_with_timeout(func, args=(), kwargs={}, timeout=5):
    result_queue = Queue()

    def target_func(result_queue, *args, **kwargs):
        try:
            # Run the function and store result in the queue
            result_queue.put(func(*args, **kwargs))
        except Exception as e:
            # Put exception in queue if it occurs
            result_queue.put(e)

    # Set up and start the process
    process = multiprocessing.Process(target=target_func, args=(result_queue, *args), kwargs=kwargs)
    process.start()
    process.join(timeout)

    # Check if the process is still alive (indicating a timeout)
    if process.is_alive():
        result = None
        if not result_queue.empty():
            result = result_queue.get()
            if isinstance(result, Exception):
                raise result  # Re-raise exception if one occurred
        process.terminate()
        process.join()
        return result
    else:
        # Check if result_queue has an exception or result
        if not result_queue.empty():
            result = result_queue.get()
            if isinstance(result, Exception):
                raise result  # Re-raise exception if one occurred
            return result
        else:
            return None


def should_perform_callback(interval: int, i: int, perform_on_zero: bool = False):

    if interval is not None:
        if perform_on_zero and i == 1:
            return True
        if i % interval == 0 and i > 0:
            return True

    return False


def rnn_callback(trainer):
    config = trainer.config
    i = trainer.iter_num

    # Check if we want to perform a callback
    if should_perform_callback(config.batch_end_callback_every, i):

        # Predict from the validation set and get the losses
        predictions = trainer.model.predict(trainer.val_dataset, sample=True)
        val_loss = predictions["total_loss"].mean().item()
        train_loss = trainer.loss.item()

        # Generate
        designs = trainer.model.generate(1000)

        # Clean designs
        designs_clean = strip_smiles(designs)
        validity, valid_smiles = smiles_validity(designs_clean, return_invalids=True)

        # Update the training history and save if a path is given in the config
        trainer.append_history(iter_num=trainer.iter_num, train_loss=train_loss, val_loss=val_loss, validity=validity)

        if trainer.outdir is not None:
            trainer.get_history(os.path.join(trainer.outdir, f"training_history.csv"))

        if wandb.run is not None:
            # Log to W&B
            wandb.log({"train_loss": train_loss,
                       "val_loss": val_loss,
                       'validity': validity,
                       'designs': designs})

        print(f"Iter: {i} ({trainer.iter_dt * 1000:.0f} ms), train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, validity: {validity:.4f}, "
              f"example: {designs[0]}")


def ae_callback(trainer):
    config = trainer.config
    i = trainer.iter_num

    # Check if we want to perform a callback
    if should_perform_callback(config.batch_end_callback_every, i):

        # Predict from the validation set and get the losses
        predictions = trainer.model.predict(trainer.val_dataset, sample=True)
        token_probs_N_S_C = predictions["token_probs_N_S_C"]
        val_loss = predictions["total_loss"].mean().item()
        target_smiles = predictions["smiles"]
        train_loss = trainer.loss.item()

        # format the designs
        designs = probs_to_smiles(token_probs_N_S_C)

        # Clean designs
        designs_clean = strip_smiles(designs)
        validity, valid_smiles = smiles_validity(designs_clean, return_invalids=True)

        # levensthein distance. This is calculated between the stripped SMILES strings. This means that if the model
        # does not learn how to place the end token, this metric is off.
        edist = np.mean([reconstruction_edit_distance(i, j) for i, j in zip(designs_clean, target_smiles)])

        # Update the training history and save if a path is given in the config
        trainer.append_history(iter_num=trainer.iter_num, train_loss=train_loss, val_loss=val_loss, validity=validity,
                               edit_distance=edist)

        if wandb.run is not None:

            try:
                # reconstruction plot
                smiles_a, smiles_b = zip(*[[target_smiles[i], valid_smiles[i]] for i, smi in enumerate(valid_smiles) if smi is not None][:4])
                edist_ab = [reconstruction_edit_distance(i, j) for i,j in zip(smiles_a, smiles_b)]
                reconstruction_plot = wandb.Image(plot_molecular_reconstruction(smiles_to_mols(smiles_a),
                                                                                smiles_to_mols(smiles_b),
                                                                                labels=edist_ab))
            except:
                reconstruction_plot = None

            tr_recon_loss = trainer.model.reconstruction_loss.mean().cpu().item()
            tr_kl_loss = trainer.model.kl_loss.mean().cpu().item() if trainer.model.kl_loss is not None else None

            # Log the grid image to W&B
            wandb.log({"train_loss": train_loss,
                       "val_loss": val_loss,
                       "train_reconstruction_loss": tr_recon_loss,
                       "train_KL_loss": tr_kl_loss,
                       'edit_distance': edist,
                       'validity': validity,
                       'designs': designs,
                       'reconstruction': reconstruction_plot})

        if trainer.outdir is not None:
            trainer.get_history(os.path.join(trainer.outdir, f"training_history.csv"))

        print(f"Iter: {i} ({trainer.iter_dt * 1000:.0f} ms), train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, validity: {validity:.4f}, "
              f"edit dist: {edist:.4f}, example: {designs[0]}, target: {target_smiles[0]}")


def mlp_callback(trainer):
    config = trainer.config
    i = trainer.iter_num

    # Check if we want to perform a callback
    if should_perform_callback(config.batch_end_callback_every, i):

        # Predict from the validation set and get the losses
        predictions = trainer.model.predict(trainer.val_dataset, sample=True)
        y_logprobs_N_K_C = predictions["y_logprobs_N_K_C"]
        target_ys = predictions["y"]
        val_loss = predictions["total_loss"].mean().item()
        train_loss = trainer.loss.item()

        # Balanced accuracy
        preds, uncertainty = logits_to_pred(y_logprobs_N_K_C, return_binary=True, return_uncertainty=True)
        b_acc = balanced_accuracy_score(preds.cpu(), target_ys.cpu())

        # Update the training history and save if a path is given in the config
        trainer.append_history(iter_num=trainer.iter_num, train_loss=train_loss, val_loss=val_loss,
                               balanced_accuracy=b_acc)

        if wandb.run is not None:

            tr_pred_loss = trainer.model.prediction_loss.mean().cpu().item()
            tr_kl_loss = trainer.model.kl_loss.mean().cpu().item() if trainer.model.kl_loss is not None else None

            # Log the grid image to W&B
            wandb.log({"train_loss": train_loss,
                       "val_loss": val_loss,
                       "train_prediction_loss": tr_pred_loss,
                       "train_KL_loss": tr_kl_loss,
                       'balanced accuracy': b_acc})

        if trainer.outdir is not None:
            trainer.get_history(os.path.join(trainer.outdir, f"training_history.csv"))

        print(f"Iter: {i} ({trainer.iter_dt * 1000:.0f} ms), train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, balanced accuracy: {b_acc:.4f}")


def jmm_callback(trainer):
    config = trainer.config
    i = trainer.iter_num

    # Check if we want to perform a callback
    if should_perform_callback(config.batch_end_callback_every, i, perform_on_zero=True):

        # Predict from the validation set
        predictions = trainer.model.predict(trainer.val_dataset, sample=True)
        y_logprobs_N_K_C = predictions["y_logprobs_N_K_C"]
        token_probs_N_S_C = predictions["token_probs_N_S_C"]
        target_ys = predictions["y"]
        target_smiles = predictions["smiles"]
        val_loss = predictions["total_loss"].mean().item()
        val_reconstruction_loss = predictions["reconstruction_loss"].mean().item()
        val_prediction_loss = predictions["prediction_loss"].mean().item()
        train_loss = trainer.loss.item()

        # Balanced accuracy
        preds, uncertainty = logits_to_pred(y_logprobs_N_K_C, return_binary=True, return_uncertainty=True)
        b_acc = balanced_accuracy_score(preds.cpu(), target_ys.cpu())

        # reconstruction
        designs = probs_to_smiles(token_probs_N_S_C)

        # Clean designs
        designs_clean = strip_smiles(designs)
        validity, valid_smiles = smiles_validity(designs_clean, return_invalids=True)

        # levensthein distance. This is calculated between the stripped SMILES strings. This means that if the model
        # does not learn how to place the end token, this metric is off.
        edist = np.mean([reconstruction_edit_distance(i, j) for i, j in zip(designs_clean, target_smiles)])

        # Update the training history and save if a path is given in the config
        trainer.append_history(iter_num=trainer.iter_num, train_loss=train_loss, val_loss=val_loss,
                               balanced_accuracy=b_acc, validity=validity, edit_distance=edist)

        if wandb.run is not None:

            try:
                # reconstruction plot
                smiles_a, smiles_b = zip(
                    *[[target_smiles[i], valid_smiles[i]] for i, smi in enumerate(valid_smiles) if smi is not None][:4])
                edist_ab = [reconstruction_edit_distance(i, j) for i, j in zip(smiles_a, smiles_b)]

                reconstruction_pil = execute_with_timeout(plot_molecular_reconstruction,
                                                          args=(smiles_to_mols(smiles_a),
                                                                smiles_to_mols(smiles_b),
                                                                edist_ab))

                reconstruction_plot = wandb.Image(reconstruction_pil)
            except:
                reconstruction_plot = None

            tr_pred_loss = trainer.model.prediction_loss.mean().cpu().item()
            tr_recon_loss = trainer.model.reconstruction_loss.mean().cpu().item()
            tr_dec_recon_loss = trainer.model.pretrained_decoder_reconstruction_loss.mean().cpu().item() if trainer.model.pretrained_decoder_reconstruction_loss is not None else None
            tr_kl_loss = trainer.model.kl_loss.mean().cpu().item() if trainer.model.kl_loss is not None else None

            # Log the grid image to W&B
            wandb.log({"train_loss": train_loss,
                       "train_prediction_loss": tr_pred_loss,
                       "train_reconstruction_loss": tr_recon_loss,
                       "train_pretrained_reconstruction_loss": tr_dec_recon_loss,
                       "train_KL_loss": tr_kl_loss,
                       "val_loss": val_loss,
                       'val_reconstruction_loss': val_reconstruction_loss,
                       'val_prediction_loss': val_prediction_loss,
                       'edit_distance': edist,
                       'balanced accuracy': b_acc, 'validity': validity, 'designs': designs,
                       'reconstruction': reconstruction_plot})

        if trainer.config.out_path is not None:
            trainer.get_history(os.path.join(config.out_path, f"training_history.csv"))

        print(f"Iter: {i} ({trainer.iter_dt * 1000:.0f} ms), train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, balanced accuracy: {b_acc:.4f}, "
              f"validity: {validity:.4f}, edit dist: {edist:.4f}, example: {designs[0]}, target: {target_smiles[0]}")

