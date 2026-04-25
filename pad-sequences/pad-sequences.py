import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    N=len(seqs)
    if max_len == None:
        max_len=max(len(seq) for seq in seqs)
    L=max_len
    result= np.full((N,L),pad_value)
    for i ,s in enumerate(seqs):
        actual_len = min(len(s), L) 
        result[i, :actual_len] = s[:actual_len]
    return result
    