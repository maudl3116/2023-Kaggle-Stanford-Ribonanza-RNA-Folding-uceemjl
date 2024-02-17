# Adapted from kaggler iafoss notebook (https://www.kaggle.com/code/iafoss/rna-starter-0-186-lb)

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import scipy
import tqdm

class RNA_Dataset(Dataset):
    def __init__(self, df, mode='train', seed=2023, fold=0, nfolds=4, 
                 mask_only=False, filter_SNR=True, **kwargs):
        self.seq_map = {'A':0,'C':1,'G':2,'U':3}                               
        self.Lmax = 206                                     # max length in training and public leaderboard
        self.filter_SNR = filter_SNR
        
        if not self.filter_SNR:
            seq_pred = pd.read_csv('/scratch/lemercier/WIP_data/test_sequences.csv')
            public = seq_pred.query('future==0')
            id_train = df.sequence_id.unique()
            id_public = public.sequence_id.unique()
            inter = list(set(id_train) & set(id_public))
        
        
        df['L'] = df.sequence.apply(len)
        df_2A3 = df.loc[df.experiment_type=='2A3_MaP']
        df_DMS = df.loc[df.experiment_type=='DMS_MaP']
        
        split = list(KFold(n_splits=nfolds, random_state=seed, 
                shuffle=True).split(df_2A3))[fold][0 if mode=='train' else 1]
        df_2A3 = df_2A3.iloc[split].reset_index(drop=True)
        df_DMS = df_DMS.iloc[split].reset_index(drop=True)
        
        if self.filter_SNR:
            
            m = (df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)    # high SNR data subset
            df_2A3 = df_2A3.loc[m].reset_index(drop=True) #.iloc[:1000]
            df_DMS = df_DMS.loc[m].reset_index(drop=True) #.iloc[:1000]
        
        else:
            # keep if good SNR or in intersection with test set. 
            m = (df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)
            m = m | (df_2A3.isin({'sequence_id': inter}).sequence_id.values) 
        
            df_2A3 = df_2A3.loc[m].reset_index(drop=True)
            df_DMS = df_DMS.loc[m].reset_index(drop=True)

        self.seq_id = df_2A3['sequence_id'].values
        self.seq_id_map = {i:self.seq_id[i] for i in range(len(self.seq_id))}
        self.seq_id_num = torch.from_numpy(np.arange(len(self.seq_id)))
        
        self.seq = df_2A3['sequence'].values                                      # all sequences
        self.L = df_2A3['L'].values                                               # all their lengths 
        
        self.react_2A3 = df_2A3[[c for c in df_2A3.columns if \
                                 'reactivity_0' in c]].values
        self.react_DMS = df_DMS[[c for c in df_DMS.columns if \
                                 'reactivity_0' in c]].values
        self.react_err_2A3 = df_2A3[[c for c in df_2A3.columns if \
                                 'reactivity_error_0' in c]].values
        self.react_err_DMS = df_DMS[[c for c in df_DMS.columns if \
                                'reactivity_error_0' in c]].values
        self.sn_2A3 = df_2A3['signal_to_noise'].values
        self.sn_DMS = df_DMS['signal_to_noise'].values
        self.mask_only = mask_only
        
        if not self.mask_only:
            self.bppms = []
            for seq_id in tqdm.tqdm(df_2A3.sequence_id.values):

                bbpm = scipy.sparse.load_npz('/scratch/lemercier/WIP_data/test/train/'+seq_id+'.npz')
#                 bbpm = scipy.sparse.load_npz('/scratch/lemercier/WIP_data/Ribonanza_bpp_files/extra_data/train/'+seq_id+'.npz')

                self.bppms.append(bbpm)
        
    def __len__(self):
        return len(self.seq)  # how many sequences
    
    def __getitem__(self, idx):
        seq = self.seq[idx]                                                        
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
        mask[:len(seq)] = True                                                     # mask above sequence length
        seq = np.pad(seq,(0,self.Lmax-len(seq)))                                   # pad sequence with zeros
        
        react = torch.from_numpy(np.stack([self.react_2A3[idx],
                                           self.react_DMS[idx]],-1))
        react_err = torch.from_numpy(np.stack([self.react_err_2A3[idx],
                                               self.react_err_DMS[idx]],-1))
        sn = torch.FloatTensor([self.sn_2A3[idx],self.sn_DMS[idx]])
                                                                                 
        return {'seq':torch.from_numpy(seq), 'bppm':torch.from_numpy(bppm).type(torch.HalfTensor), 'mask':mask},\
               {'react':react.type(torch.FloatTensor), 'react_err':react_err.type(torch.FloatTensor),
                'sn':sn, 'mask':mask, 'seq_id_num':self.seq_id_num[idx]}
    
class LenMatchBatchSampler(torch.utils.data.BatchSampler):               
    def __iter__(self):
        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            s = self.sampler.data_source[idx]                           # of the form {'mask':mask}, {'mask':mask} 
            if isinstance(s,tuple): L = s[0]["mask"].sum()              
            else: L = s["mask"].sum()
            L = max(1,L // 16)                                         
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
            
#==============================================================================================================================                

class RNA_Sub_Dataset(Dataset):
    def __init__(self, df, seed=2023, 
                 mask_only=False, **kwargs):
        self.seq_map = {'A':0,'C':1,'G':2,'U':3}                                 
        self.Lmax = 206                                                           # max length in submission dataset
#         df['L'] = df.sequence.apply(lambda x: len(x))
   
        self.seq = df['sequence'].values                                      # all sequences
#         self.L = df['L'].values                                               # all their lengths 
        
        self.id_min = df['id_min'].values
        self.id_max = df['id_max'].values

        self.mask_only = mask_only
                
        if not self.mask_only:
            self.bppms = []
            for seq_id in tqdm.tqdm(df.sequence_id.values):

                bbpm = scipy.sparse.load_npz('/scratch/lemercier/WIP_data/Ribonanza_bpp_files/extra_data/test/'+seq_id+'.npz')
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
            
# Example usage


# ds_train = RNA_Dataset(df, mode='train', fold=fold, nfolds=nfolds)
# ds_train_len = RNA_Dataset(df, mode='train', fold=fold, 
#             nfolds=nfolds, mask_only=True)
# sampler_train = torch.utils.data.RandomSampler(ds_train_len)
# len_sampler_train = LenMatchBatchSampler(sampler_train, batch_size=bs,
#             drop_last=True)
# dl_train = DeviceDataLoader(torch.utils.data.DataLoader(ds_train, 
#             batch_sampler=len_sampler_train, num_workers=num_workers,
#             persistent_workers=True), device)

# ds_val = RNA_Dataset(df, mode='eval', fold=fold, nfolds=nfolds)
# ds_val_len = RNA_Dataset(df, mode='eval', fold=fold, nfolds=nfolds, 
#            mask_only=True)
# sampler_val = torch.utils.data.SequentialSampler(ds_val_len)
# len_sampler_val = LenMatchBatchSampler(sampler_val, batch_size=bs, 
#            drop_last=False)
# dl_val= DeviceDataLoader(torch.utils.data.DataLoader(ds_val, 
#            batch_sampler=len_sampler_val, num_workers=num_workers), device)
# gc.collect()

# data = DataLoaders(dl_train,dl_val)