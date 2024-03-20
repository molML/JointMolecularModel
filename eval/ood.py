
from torch.utils.data.dataloader import DataLoader
from jcm.utils import to_binary, ClassificationMetrics, logits_to_pred, single_batchitem_fix
from dataprep.utils import tanimoto_matrix, mols_to_scaffolds, smiles_to_mols
from dataprep.descriptors import mols_to_ecfp
import numpy as np
import torch
from tqdm import tqdm


def ood_pred(model, config, dataset):

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

    assert all(data['y'].clone().detach() == torch.tensor(dataset.y)), 'Samples and labels do not match'

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
