import torch
import esm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
model = model.to(device)

batch_converter = alphabet.get_batch_converter()
model.eval()

import pandas as pd

train_data = pd.read_csv("test.csv")

X = train_data[["id", "sequence"]]

X_tuple = list(X.itertuples(index=False, name=None))

batched_labels = []
batched_strs = []
batched_tokens = []
batched_lens = []
n = 4

for i in range(0, len(X_tuple), n):
  batch_labels, batch_strs, batch_tokens = batch_converter(X_tuple[i:i+n])
  batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
  batched_labels.append(batch_labels)
  batched_strs.append(batch_strs)
  batched_tokens.append(batch_tokens)
  batched_lens.append(batch_lens)
#print([batched_tokens[i].shape for i in range(len(batched_tokens))])

num_batches = len(batched_tokens)

all_sequence_representations = ()

for batch in range(num_batches):
    print("Batch {}".format(batch))
    with torch.no_grad():
        print(batched_tokens[batch].shape)
        results = model(batched_tokens[batch].to(device), repr_layers=[6], return_contacts=False)
    token_representations = results["representations"][6]
    print(token_representations.shape)

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batched_lens[batch].tolist()):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1, :].mean(0).unsqueeze(dim=0))
    all_sequence_representations = all_sequence_representations + tuple(sequence_representations)

all_sequence_representations = torch.cat(all_sequence_representations, 0)

print(all_sequence_representations.shape)
torch.save(all_sequence_representations, "test_seq_reps.pt")
