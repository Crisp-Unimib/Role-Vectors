import os
import json

dataset_dir_path = os.path.dirname(os.path.realpath(__file__))

SPLITS = ['train', 'val', 'val_processed', 'test',  
          'train_archaeologist', 
          'train_archivist', 
          'train_bailiff',
          'train_biologist',
          'train_chemist', 
          'train_data analyst',
          'train_data scientist',
          'train_dentist',
            'train_doctor',
            'train_ecologist',
            'train_economic researcher',
            'train_economist',
            'train_electrical engineer',
            'train_electronics technician',
            'train_enthusiast',
            'train_financial analyst',
            'train_geneticist',
            'train_historian',
            'train_historical researcher',
            'train_lawyer',
            'train_mathematician',
            'train_nurse',
            'train_partisan',
            'train_physician',
            'train_physicist',
            'train_politician',
            'train_psychologist',
            'train_sheriff',
            'train_software engineer',
            'train_statistician',
            'train_surgeon',
            'train_teacher',
            'train_web developer',

<<<<<<< HEAD
          'test_natural_science', 'test_law', 'test_econ', 'test_eecs', 'test_math', 'test_medicine', 'test_natural_science', 'test_politics', 'test_psychology', 'test_aime']
=======
          'test_natural_science', 'test_law', 'test_econ', 'test_eecs', 'test_math', 'test_medicine', 'test_natural_science', 'test_politics', 'test_psychology', 'test_new_medicine']
>>>>>>> updates
DATA_VARIANTS = ['base', 'target', 'mmlu']

SPLIT_DATASET_FILENAME = os.path.join(dataset_dir_path, 'splits/{type}_{split}.json')

PROCESSED_DATASET_NAMES = ["chembench", "alpaca", "test"]

def load_dataset_split(datavar: str, split: str, instructions_only: bool=False):
    assert datavar in DATA_VARIANTS
    assert split in SPLITS

    file_path = SPLIT_DATASET_FILENAME.format(type=datavar, split=split)

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['instruction'] for d in dataset]

    return dataset

def load_dataset(dataset_name, instructions_only: bool=False):
    assert dataset_name in PROCESSED_DATASET_NAMES, f"Valid datasets: {PROCESSED_DATASET_NAMES}"

    file_path = os.path.join(dataset_dir_path, 'processed', f"{dataset_name}.json")

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['instruction'] for d in dataset]
 
    return dataset