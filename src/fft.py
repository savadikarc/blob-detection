import numpy as np

def DFT2(f):
    
    R, C = f.shape
    
    # DFT of rows
    rows_dft = []
    for i in range(R):
        row_dft = np.fft.fft(f[i, :])
        row_dft = np.expand_dims(row_dft, axis=0)
        rows_dft.append(row_dft)
    rows_dft = np.concatenate(rows_dft, axis=0)
    
    # DFT of the columns of the result
    dft = []
    for j in range(C):
        col_dft = np.fft.fft(rows_dft[:, j])
        col_dft = np.expand_dims(col_dft, axis=1)
        dft.append(col_dft)
    dft = np.concatenate(dft, axis=1)
    return dft

def IDFT2(F):
    
    swapped = F.imag + complex("j")*F.real
    dft_swapped = DFT2(swapped)
    swapped_dft = dft_swapped.imag + complex("j")*dft_swapped.real
    
    return swapped_dft / (swapped_dft.shape[0]*swapped_dft.shape[1])


def shift_indices(dim):
    """
    Returns the order of indices which shifts the dft to a centered spectrum
    """
    
    if (dim % 2) == 0:
        positive_freq = np.arange(1, dim//2)
        negative_freq = np.arange(dim//2, dim)
        zero_freq = np.array([0])
    else:
        positive_freq = np.arange(1, (dim-1)//2+1)
        negative_freq = np.arange((dim-1)//2+1, dim)
        zero_freq = np.array([0])
    indices = np.concatenate([negative_freq, zero_freq, positive_freq])
    return indices

def fft_shift(dft):
    
    R, C = dft.shape
    
    row_indices = shift_indices(R)
    col_indices = shift_indices(C)
    
    return dft[row_indices, :][:, col_indices]