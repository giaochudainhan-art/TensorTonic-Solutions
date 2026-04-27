import numpy as np
import numpy as np
def dropout(x, p=0.5, rng=None):
    x = np.array(x, dtype=np.float64)
    keep_prob = 1 - p
    
    if rng is not None:
        # Quan trọng: Dùng rng.random(x.shape) 
        # để khớp với seed 123 của hệ thống
        random_values = rng.random(x.shape)
    else:
        random_values = np.random.random(x.shape)
        
    # Bước quyết định: So sánh với keep_prob (1-p)
    dropout_pattern = random_values >= p  # Thay vì < keep_prob
    
    # Tính toán: x * mask rồi mới chia scale
    output = (x * dropout_pattern) / keep_prob
    
    return (output, dropout_pattern.astype(int)/keep_prob)

