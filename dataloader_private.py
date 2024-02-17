import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import scipy
import tqdm

#TODO: merge with dataloader.py as only some paths and Lmax are changing. 

class LenMatchBatchSampler(torch.utils.data.BatchSampler):             
    def __iter__(self):
        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            s = self.sampler.data_source[idx]                           # of the form {'mask':mask},{'mask':mask} 
            if isinstance(s,tuple): L = s[0]["mask"].sum()              
            else: L = s["mask"].sum()
            L = max(1,L // 16)                                          # what is this ??? because minimum bs is expected 16?
            if len(buckets[L]) == 0:  buckets[L] = []
            buckets[L].append(idx)
            
            if len(buckets[L]) == self.batch_size:
                batch = list(buckets[L])
                yield batch
                yielded += 1
                buckets[L] = []
                
        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch
            
def dict_to(x, device='cuda'):
    return {k:x[k].to(device) for k in x}

def to_device(x, device='cuda'):
    return tuple(dict_to(e,device) for e in x)

class DeviceDataLoader:
    def __init__(self, dataloader, device='cuda'):
        self.dataloader = dataloader
        self.device = device
    
    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(dict_to(x, self.device) for x in batch)
            

class RNA_Sub_Dataset(Dataset):
    def __init__(self, df, seed=2023, 
                 mask_only=False, **kwargs):
        self.seq_map = {'A':0,'C':1,'G':2,'U':3}                                  # one hot encoding
        self.Lmax = 457                                                           # max length in private LB
#         df['L'] = df.sequence.apply(lambda x: len(x))
     
        self.seq = df['sequence'].values                                        # all sequences
#         self.L = df['L'].values                                               # all their lengths 
        
        self.id_min = df['id_min'].values
        self.id_max = df['id_max'].values

        self.mask_only = mask_only
                
        if not self.mask_only:
            self.bppms = []
            for seq_id in tqdm.tqdm(df.sequence_id.values):
                try:
                    bbpm = scipy.sparse.load_npz('/scratch/lemercier/private/'+seq_id+'.npz')
                except:
                    bbpm = scipy.sparse.load_npz('/scratch/lemercier/WIP_data/test/'+seq_id+'.npz')
                self.bppms.append(bbpm)
        
    def __len__(self):
        return len(self.seq)  # how many sequences
    
    def __getitem__(self, idx):
        seq = self.seq[idx]                                                       
        id_min = self.id_min[idx] 
        id_max = self.id_max[idx] 
        if self.mask_only:                                                        
            mask = torch.zeros(self.Lmax, dtype=torch.bool)
            mask[:len(seq)] = True
            return {'mask':mask},{'mask':mask}                                     
        seq = [self.seq_map[s] for s in seq]                                       
        seq = np.array(seq)
        
        
        bppm = self.bppms[idx].toarray()
        #pad
        dim1, dim2 = bppm.shape[0], bppm.shape[1]
        bppm = np.pad(bppm,((0,self.Lmax-dim1),(0,self.Lmax-dim2)) ) 
        #make sym
        bppm = bppm+bppm.T
        
        mask = torch.zeros(self.Lmax, dtype=torch.bool)
        mask[:len(seq)] = True                                                     
        seq = np.pad(seq,(0,self.Lmax-len(seq)))                                  
            
        return {'seq':torch.from_numpy(seq), 'bppm':torch.from_numpy(bppm).type(torch.HalfTensor), 'mask':mask, 'id_min':id_min, 'id_max':id_max},\
               {'mask':mask}            
            
