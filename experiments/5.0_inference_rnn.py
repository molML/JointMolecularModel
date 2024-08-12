""" Perform inference of all molecules on the RNN control model

Derek van Tilborg
Eindhoven University of Technology
Augustus 2024
"""

import os
from os.path import join as ospj
import pandas as pd
from constants import ROOTDIR
from jcm.config import Config, load_settings, load_and_setup_config_from_file


if __name__ == "__main__":

    RNN_RESULTS = ospj('results', 'rnn_pretraining')

    # move to root dir
    os.chdir(ROOTDIR)

    # 1. Load the model

    # find the best pretrained model based on validation loss
    best_rows_per_exp = []
    experiment_dirs = [i for i in os.listdir(RNN_RESULTS) if not i.startswith('.')]
    for exp_name in experiment_dirs:

        # load training history file
        df = pd.read_csv(ospj(RNN_RESULTS, exp_name, 'training_history.csv'))
        df['experiment'] = exp_name

        # Select the row with the minimum value in column 'A'
        min_value_row = df.loc[df['val_loss'].idxmin()]
        best_rows_per_exp.append(dict(min_value_row))

    pretraining_results = pd.DataFrame(best_rows_per_exp).set_index('experiment')

    best_experiment = pretraining_results['val_loss'].idxmin()

    # load the model settings
    config = load_and_setup_config_from_file(ospj(RNN_RESULTS, best_experiment, 'experiment_settings.yml'))







    # 2. Get all data
    # 3. Perform inference
    # 4. Save results.




