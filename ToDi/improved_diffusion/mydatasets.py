from torch.utils.data import DataLoader, Dataset
import torch
import random
from rdkit import Chem
from rdkit import RDLogger
from torch.utils.data import DistributedSampler
RDLogger.DisableLog('rdApp.*')
import csv
import random
import torch
from torch.utils.data import Dataset

def get_dataloader(dataset, batchsize, rank, world_size):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    def collate(batch):
        toked_selfies = [i['tok_selfies'] for i in batch]
        desc_states = [i['desc_state'] for i in batch]
        desc_mask = [i['desc_mask'] for i in batch]
        corrupted_toked_selfies = [i['corrupted_toked_selfies'] for i in batch]
        gene_expression = [i['gene_expression'] for i in batch]

        return torch.concat(toked_selfies, dim=0), torch.concat(desc_states, dim=0), torch.concat(desc_mask, dim=0), torch.concat(corrupted_toked_selfies, dim=0), torch.concat(gene_expression, dim=0)

    dataloader = DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=False,
        collate_fn=collate,
        sampler=sampler
    )

    def cycle():
        ec = 0
        while True:
            dataloader.sampler.set_epoch(ec)
            for i in dataloader:
                yield i
            ec += 1

    return iter(cycle())


class SELFIESdataset(Dataset):
    def __init__(self, dir, selfies_tokenizer, split, replace_desc=False, pre=None, prob=0, load_state=True,
                 corrupt_prob=0.0, mask_desc=False):
        super().__init__()
        self.dir = dir
        self.selfies_tokenizer = selfies_tokenizer
        self.split = split
        self.replace_desc = replace_desc
        self.pre = pre
        self.prob = prob
        self.corrupt_prob = corrupt_prob
        print('corruption prob is {}'.format(self.corrupt_prob))
        self.mask_desc = mask_desc
        print('mask_desc is {}'.format(self.mask_desc))
        assert split in ['train', 'test', 'validation', 'train_val_256', 'AKT1', 'AKT2', 'AURKB', 'CTSK', 'EGFR', 'HDAC1', 'MTOR', 'PIK3CA', 'SMAD3', 'TP53']
        self.ori_data = self.get_ori_data()
        self.load_state = load_state
        if load_state:
            self.desc_state = self.get_desc_state()

    def get_desc_state(self):
        import os.path as osp
        file_path = osp.join(self.dir, self.split + '_desc_states_256.pt')
        return torch.load(file_path)

    def get_ori_data(self):
        import os.path as osp
        res = []
        file_path = osp.join(self.dir, self.split + '.csv')

        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for i, line in enumerate(reader):
                gene_id = line[0].strip()
                selfies = line[2].strip()
                if len(line) > 3:
                    desc = line[3].strip()
                else:
                    desc = ""

                gene_expression = [float(x) for x in line[4:]]

                if self.replace_desc:
                    import spacy
                    nlp = spacy.load('en_core_web_sm')
                    doc = nlp(desc)
                    for token in doc:
                        if token.text == 'is':
                            desc = 'The molecule ' + desc[token.idx:]
                            break

                res.append((gene_id, selfies, desc, gene_expression))

        return res

    def __len__(self):
        return len(self.ori_data)

    def permute(self, selfies):
        p = random.random()
        if p < self.prob:
            print("PERMUTE SELFIES")
            return changeorder(selfies, shuffle=True)
        else:
            return selfies

    def __getitem__(self, idx):
        data = self.ori_data[idx]
        dic = {'cid': data[0], 'selfies': self.permute(data[1]), 'desc': data[2], 'gene_expression': data[3]}
        dic['gene_expression'] = torch.tensor(dic['gene_expression'], dtype=torch.float32).unsqueeze(0)
        dic['tok_selfies'] = self.selfies_tokenizer(dic['selfies'])
        dic['corrupted_toked_selfies'] = self.selfies_tokenizer.corrupt(
            dic['selfies']) if random.random() < self.corrupt_prob else dic['tok_selfies']
        dic['tok_desc'] = None
        dic['desc_mask'] = None
        if self.load_state:
            dic['desc_state'] = self.desc_state[data[0]]['states']
            dic['desc_mask'] = self.desc_state[data[0]]['mask']
            if self.mask_desc:
                dic['desc_state'] = torch.zeros_like(dic['desc_state'])
                dic['desc_mask'] = torch.ones_like(dic['desc_mask'])
        return dic


def changeorder(selfies, shuffle):
    original_selfies = selfies
    mol = Chem.MolFromSmiles(original_selfies)
    if mol is None:
        print("Wrong in original dataset")
    Chem.Kekulize(mol)
    atom_indices = [atom.GetIdx() for atom in mol.GetAtoms()]
    if shuffle:
        random.shuffle(atom_indices)
    reordered_mol = Chem.RenumberAtoms(mol, atom_indices)
    new_selfies = Chem.MolToSmiles(reordered_mol, kekuleSmiles=True)
    return new_selfies
