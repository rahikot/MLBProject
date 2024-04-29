import torch
import esm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
model = model.to(device)

batch_converter = alphabet.get_batch_converter()
model.eval()

import pandas as pd
import os

files  = ['Wu_test.csv',
 'Olson_test.csv',
 'Chen_test.csv',
 'Pokusaeva_train.csv',
 'Tsuboyama_train.csv',
 'Tsuboyama_test.csv',
 'Wu_train.csv',
 'Sinai_train.csv']

from tqdm import tqdm, trange

for file in tqdm(files):
    train_data = pd.read_csv(os.path.join('combined', file))

    X = train_data[["mutant", "mutated_sequence"]]

    X_tuple = list(X.itertuples(index=False, name=None))

    batched_labels = []
    batched_strs = []
    batched_tokens = []
    batched_lens = []
    n = 64

    for i in trange(0, len(X_tuple), n):
      batch_labels, batch_strs, batch_tokens = batch_converter(X_tuple[i:i+n])
      batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
      batched_labels.append(batch_labels)
      batched_strs.append(batch_strs)
      batched_tokens.append(batch_tokens)
      batched_lens.append(batch_lens)
    #print([batched_tokens[i].shape for i in range(len(batched_tokens))])

    num_batches = len(batched_tokens)

    all_sequence_representations = ()

    for batch in trange(num_batches):
        with torch.no_grad():
            #print(batched_tokens[batch].shape)
            results = model(batched_tokens[batch].to(device), repr_layers=[6], return_contacts=False)
        token_representations = results["representations"][6]
        #print(token_representations.shape)

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        sequence_representations = []
        for i, tokens_len in enumerate(batched_lens[batch].tolist()):
            sequence_representations.append(token_representations[i, 1 : tokens_len - 1, :].mean(0).unsqueeze(dim=0))
        all_sequence_representations = all_sequence_representations + tuple(sequence_representations)

    all_sequence_representations = torch.cat(all_sequence_representations, 0)

    #print(all_sequence_representations.shape)
    torch.save(all_sequence_representations, os.path.join("embeddings", os.path.splitext(file)[0] + "_seq.pt"))
