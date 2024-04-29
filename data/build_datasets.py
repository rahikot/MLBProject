import torch
import pandas as pd
from torch.utils.data import TensorDataset
import os
from tqdm import tqdm

names = ['Chen_train',
 'Pokusaeva_test',
 'Olson_train',
 'Sinai_test',
 'Wu_test',
 'Olson_test',
 'Chen_test',
 'Pokusaeva_train',
 'Tsuboyama_train',
 'Tsuboyama_test',
 'Wu_train',
 'Sinai_train']

for name in tqdm(names):
    embeddings_tensor = torch.load(os.path.join('embeddings', name + '_seq.pt'))
    df = pd.read_csv(os.path.join('combined', name + '.csv'))
    targets = torch.Tensor(df['DMS_score']).unsqueeze(-1)
    name_ds = TensorDataset(embeddings_tensor, targets)
    torch.save(name_ds, os.path.join('datasets', name + '.pt'))

