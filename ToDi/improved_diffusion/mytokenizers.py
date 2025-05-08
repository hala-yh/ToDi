import torch
import os
from os import path as osp
import regex
import random

def getrandomnumber(numbers, k, weights=None):
    if k == 1:
        return random.choices(numbers, weights=weights, k=k)[0]
    else:
        return random.choices(numbers, weights=weights, k=k)

class regexTokenizer():
    def __init__(self, path='.../selfies_vocab.txt', max_len=258):
        print('Truncating length:', max_len)
        with open(path, 'r') as f:
            x = f.readlines()

        pattern = r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]|Branch[0-9]+|Ring[0-9]+)"
        self.rg = regex.compile(pattern)
        self.idtotok = {cnt + 3: i.strip() for cnt, i in enumerate(x)}
        self.idtotok.update({
            0: '[PAD]',
            1: '[SOS]',
            2: '[EOS]',
            3: '[Si]',
            4: '.'
        })
        self.vocab_size = len(self.idtotok)
        self.toktoid = {v: k for k, v in self.idtotok.items()}
        self.max_len = max_len

    def decode_one(self, iter):
        return "".join([self.idtotok[i.item()] for i in iter])

    def decode(self, ids: torch.tensor):
        if len(ids.shape) == 1:
            return [self.decode_one(ids)]
        else:
            selfies = []
            for i in ids:
                selfies.append(self.decode_one(i))
            return selfies

    def __len__(self):
        return self.vocab_size

    def __call__(self, selfies: list):
        tensors = []
        if type(selfies) is str:
            selfies = [selfies]
        for i in selfies:
            tensors.append(self.encode_one(i))
        return torch.concat(tensors, dim=0)

    def encode_one(self, selfie):
        res = [self.toktoid[i] for i in self.rg.findall(selfie)]
        res = [1] + res + [2]
        if len(res) < self.max_len:
            res += [0] * (self.max_len - len(res))
        else:
            res = res[:self.max_len]
            res[-1] = 2
        return torch.LongTensor([res])

    def corrupt(self, selfies: list):
        tensors = []
        if type(selfies) is str:
            selfies = [selfies]
        for i in selfies:
            tensors.append(self.corrupt_one(i))
        return torch.concat(tensors, dim=0)

    def corrupt_one(self, selfie):
        res = [i for i in self.rg.findall(selfie)]
        total_length = len(res) + 2
        if total_length > self.max_len:
            return self.encode_one(selfie)

        r = random.random()
        if r < 0.3:
            pa, ring = True, True
        elif r < 0.65:
            pa, ring = True, False
        else:
            pa, ring = False, True

        max_ring_num = 1
        ringpos = []
        papos = []
        for pos, at in enumerate(res):
            if at == '(' or at == ')':
                papos.append(pos)
            elif at.isnumeric():
                max_ring_num = max(max_ring_num, int(at))
                ringpos.append(pos)

        r = random.random()
        if r < 0.3:
            remove, padd = True, True
        elif r < 0.65:
            remove, padd = True, False
        else:
            remove, padd = False, True
        if pa and len(papos) > 0:
            if remove:
                n_remove = random.choice([1, 2, 3, 4])
                p_remove = set(random.choices(papos, k=n_remove))
                total_length -= len(p_remove)
                for p in p_remove:
                    res[p] = None

        r = random.random()
        if r < 0.3:
            remove, radd = True, True
        elif r < 0.65:
            remove, radd = True, False
        else:
            remove, radd = False, True
        if ring and len(ringpos) > 0:
            if remove:
                n_remove = random.choice([1, 2, 3, 4])
                p_remove = set(random.choices(ringpos, k=n_remove))
                total_length -= len(p_remove)
                for p in p_remove:
                    res[p] = None

        if pa and padd:
            n_add = random.choice([1, 2, 3])
            n_add = min(self.max_len - total_length, n_add)
            for _ in range(n_add):
                sele = random.randrange(len(res) + 1)
                res.insert(sele, '(' if random.random() < 0.5 else ')')
                total_length += 1

        if ring and radd:
            n_add = random.choice([1, 2, 3])
            n_add = min(self.max_len - total_length, n_add)
            for _ in range(n_add):
                sele = random.randrange(len(res) + 1)
                res.insert(sele, str(random.randrange(1, max_ring_num + 1)))
                total_length += 1

        res = [self.toktoid[i] for i in res if i is not None]
        res = [1] + res + [2]
        if len(res) < self.max_len:
            res += [0] * (self.max_len - len(res))
        else:
            res = res[:self.max_len]
            res[-1] = 2
        return torch.LongTensor([res])
