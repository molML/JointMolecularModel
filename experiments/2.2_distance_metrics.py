
from jcm.training_logistics import get_all_dataset_names, load_dataset_df
from constants import ROOTDIR
import os





if __name__ == '__main__':

    # move to root dir
    os.chdir(ROOTDIR)

    all_datasets = get_all_dataset_names()


