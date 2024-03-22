
from torch.utils.data.dataloader import DataLoader
from jcm.utils import ClassificationMetrics, reconstruction_metrics, single_batchitem_fix, logits_to_pred
from dataprep.utils import tanimoto_matrix, mols_to_scaffolds, smiles_to_mols
from dataprep.descriptors import mols_to_ecfp
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore", message="An input array is constant; the correlation coefficient is not defined.")



def pred(model, config, dataset):

    test_loader = DataLoader(dataset, shuffle=False, pin_memory=True, batch_size=config.batch_size,
                             collate_fn=single_batchitem_fix)

    data = {'y_logits_N_K_C': [], 'x_hat': [], 'z': [], 'sample_likelihood': [], 'y': [], 'x': []}

    model.eval()
    for batch in tqdm(test_loader):

        x_, y_ = batch
        x_.to(config.device)
        y_.to(config.device)

        y_logits_N_K_C_, x_hat_, z_, sample_likelihood_, loss_ = model(x_, y_)

        data['y_logits_N_K_C'].append(y_logits_N_K_C_)
        data['x_hat'].append(x_hat_)
        data['z'].append(z_)
        data['sample_likelihood'].append(sample_likelihood_)
        data['y'].append(y_)
        data['x'].append(x_)

    model.train()

    # concatenate all entries
    for k, v in data.items():
        data[k] = torch.cat(v)

    assert all(data['y'].clone().detach() == dataset.y.clone().detach()), 'Samples and labels do not match'

    y_hat, y_hat_uncertainty = logits_to_pred(data['y_logits_N_K_C'], return_binary=True)
    data['y_hat'] = y_hat
    data['y_hat_uncertainty'] = y_hat_uncertainty

    return data


def test_train_sim(test_smiles: list[str], train_smiles: list[str], scaffold: bool = False):

    all_mols = smiles_to_mols(train_smiles + test_smiles)
    if scaffold:
        all_mols = mols_to_scaffolds(all_mols)

    M = tanimoto_matrix(mols_to_ecfp(all_mols))

    test_train_sims = M[len(train_smiles):, :len(train_smiles)]

    tani_mean = np.mean(test_train_sims, 1)  # 0.1238, while its 0.1417 for the within similarity
    tani_max = np.max(test_train_sims, 1)
    tani_min = np.min(test_train_sims, 1)

    return tani_mean, tani_max, tani_min


def distance_to_trainset(test_dataset, train_dataset):

    tani_mean, tani_max, tani_min = test_train_sim(test_dataset.smiles, train_dataset.smiles)

    test_dataset.tani_mean = tani_mean
    test_dataset.tani_max = tani_max
    test_dataset.tani_min = tani_min

    scaff_tani_mean, scaff_tani_max, scaff_tani_min = test_train_sim(test_dataset.smiles,
                                                                     train_dataset.smiles, scaffold=True)

    test_dataset.scaff_tani_mean = scaff_tani_mean
    test_dataset.scaff_tani_max = scaff_tani_max
    test_dataset.scaff_tani_min = scaff_tani_min

    return test_dataset


def ood_correlations(likelihoods, y_hat_uncertainty, test_dataset):
    """ Computes the correlation between data distances and something else

    :param likelihoods:
    :param y_hat_uncertainty:
    :param test_dataset:
    :return:
    """

    metrics = {}

    # Correlations between likelihoods and distances
    metrics['corr_likelihood_medoid_dist'] = pearsonr(likelihoods, 1 - np.array(test_dataset.sim_to_train_medoid)).correlation
    metrics['corr_likelihood_tani_mean'] = pearsonr(likelihoods, 1 - test_dataset.tani_mean).correlation
    metrics['corr_likelihood_tani_max'] = pearsonr(likelihoods, 1 - test_dataset.tani_max).correlation
    metrics['corr_likelihood_tani_min'] = pearsonr(likelihoods, 1 - test_dataset.tani_min).correlation
    metrics['corr_likelihood_scaff_tani_mean'] = pearsonr(likelihoods, 1 - test_dataset.scaff_tani_mean).correlation
    metrics['corr_likelihood_scaff_tani_max'] = pearsonr(likelihoods, 1 - test_dataset.scaff_tani_max).correlation
    metrics['corr_likelihood_scaff_tani_min'] = pearsonr(likelihoods, 1 - test_dataset.scaff_tani_min).correlation

    # Correlations between y_hat uncertainties and distances
    metrics['corr_uncertainty_medoid_dist'] = pearsonr(y_hat_uncertainty, 1 - np.array(test_dataset.sim_to_train_medoid)).correlation
    metrics['corr_uncertainty_tani_mean'] = pearsonr(y_hat_uncertainty, 1 - test_dataset.tani_mean).correlation
    metrics['corr_uncertainty_tani_max'] = pearsonr(y_hat_uncertainty, 1 - test_dataset.tani_max).correlation
    metrics['corr_uncertainty_tani_min'] = pearsonr(y_hat_uncertainty, 1 - test_dataset.tani_min).correlation
    metrics['corr_uncertainty_scaff_tani_mean'] = pearsonr(y_hat_uncertainty, 1 - test_dataset.scaff_tani_mean).correlation
    metrics['corr_uncertainty_scaff_tani_max'] = pearsonr(y_hat_uncertainty, 1 - test_dataset.scaff_tani_max).correlation
    metrics['corr_uncertainty_scaff_tani_min'] = pearsonr(y_hat_uncertainty, 1 - test_dataset.scaff_tani_min).correlation

    return metrics


def evaluate_predictions(model, config, test_dataset, train_dataset):
    results = {}

    test_dataset = distance_to_trainset(test_dataset, train_dataset)
    data = pred(model, config, test_dataset)

    correlations = ood_correlations(likelihoods=data['sample_likelihood'].tolist(),
                                    y_hat_uncertainty=data['y_hat_uncertainty'].tolist(),
                                    test_dataset=test_dataset)

    metrics_y = {'y_' + k: v for k, v in ClassificationMetrics(y=data['y'], y_hat=data['y_hat']).all().items()}
    metrics_x = {'x_' + k: v for k, v in reconstruction_metrics(x=data['x'], x_hat=data['x_hat']).items()}

    results['mean_uncertainty_y'] = torch.mean(data['y_hat_uncertainty']).item()
    results['mean_sample_likelihood'] = torch.mean(data['sample_likelihood']).item()
    results = results | metrics_y | metrics_x | correlations

    return results
