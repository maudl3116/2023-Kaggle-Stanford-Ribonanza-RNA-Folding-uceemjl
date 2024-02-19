# 2023-Kaggle-Stanford-Ribonanza-RNA-Folding-uceemjl

Builds upon the "best single model" shared by the kaggler iafoss. 
Substantial performance improvement (public leaderboard position) was obtained by:
  - extracting base pairing probability matrices (bppm) predicted with contrafold2 to bias the attention using the bppm.
  - replacing the sinusoidal positional encoding with a relative positional encoding.
