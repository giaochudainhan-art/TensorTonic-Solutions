import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
import numpy as np

def adam_step(param, grad, m, v, i, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Thực hiện MỘT bước cập nhật trọng số theo thuật toán Adam.
    i: bước thứ mấy (bắt đầu từ 1)
    """
    # 1. Ép kiểu NumPy để tính toán ma trận
    param = np.asarray(param)
    grad = np.asarray(grad)
    m = np.asarray(m)
    v = np.asarray(v)

    # 2. Cập nhật Moment bậc 1 và bậc 2 (Tích lũy lịch sử)
    # Chúng ta cập nhật trực tiếp vào m và v gốc
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad**2)

    # 3. Bias Correction (Tính giá trị hiệu chỉnh - Dùng biến tạm)
    # QUAN TRỌNG: Không được ghi đè lên m và v gốc ở đây
    m_hat = m / (1 - beta1**i)
    v_hat = v / (1 - beta2**i)

    # 4. Cập nhật trọng số
    param = param - lr * m_hat / (np.sqrt(v_hat) + eps)

    return param, m, v