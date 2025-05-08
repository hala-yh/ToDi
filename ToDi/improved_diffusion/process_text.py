from mydatasets import get_dataloader, SELFIESdataset
import torch
from transformers import AutoModel, AutoTokenizer
from mytokenizers import regexTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True)
args = parser.parse_args()
split = args.input

selfies_tokenizer = regexTokenizer()

train_dataset = SELFIESdataset(
    dir='.../DATA/',
    selfies_tokenizer=selfies_tokenizer,
    split=split,
    replace_desc=False,
    load_state=False
)

model = AutoModel.from_pretrained('.../scibert')
tokz = AutoTokenizer.from_pretrained('.../scibert')

volume = {}

model = model.cuda()
model.eval()

with torch.no_grad():
    for i in range(len(train_dataset)):
        if i % 190 == 0:
            print(i)

        id = train_dataset[i]['cid']
        desc = train_dataset[i]['desc']

        tok_op = tokz(
            desc,
            max_length=256,
            truncation=True,
            padding='max_length'
        )
        toked_desc = torch.tensor(tok_op['input_ids']).unsqueeze(0)
        toked_desc_attentionmask = torch.tensor(tok_op['attention_mask']).unsqueeze(0)

        assert toked_desc.shape[1] == 256

        lh = model(toked_desc.cuda()).last_hidden_state

        volume[id] = {
            'states': lh.cpu(),
            'mask': toked_desc_attentionmask
        }

torch.save(volume, '.../DATA/' + split + '_desc_states_256.pt')