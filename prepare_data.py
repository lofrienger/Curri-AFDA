from pathlib import Path

data_path_root = Path('/home/ren2/data2/wa_dataset/Fundus_ROIs/')

def get_data_paths_list(domain='Domain1', split='train', type='image'):
    paths_list = list((data_path_root / domain / split / type).glob('*'))
    return paths_list
