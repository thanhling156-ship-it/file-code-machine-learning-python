import matplotlib.pyplot as plt
import numpy as np

# --- 1. Chuẩn bị dữ liệu ---
n_samples_train = 100
n_samples_test = 40
center = np.array([0, 0])
sigma1 = 0.2
sigma2 = 0.6

X1 = np.random.randn(n_samples_train, 2) * sigma1 + center
X2 = np.random.randn(n_samples_train, 2) * sigma2 + center

X1_test = np.random.randn(n_samples_test, 2) * sigma1 + center
X2_test = np.random.randn(n_samples_test, 2) * sigma2 + center

X = np.vstack((X1, X2))
# Tổng số mẫu là N = 200
N = X.shape[0] 
Y = np.vstack((np.zeros((n_samples_train, 1)), np.ones((n_samples_train, 1))))

# --- 2. Khởi tạo Trọng số và Bias ---
W1 = np.random.randn(2, 100) * 0.1
b1 = np.zeros((1, 100)) # Thêm bias b1
W2 = np.random.randn(100, 1) * 0.1
b2 = np.zeros((1, 1))   # Thêm bias b2
lr = 0.1

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for i in range(1001):
    # --- Forward ---
    Z1 = X @ W1 + b1 # Thêm + b1
    H = sigmoid(Z1)
    
    Z2 = H @ W2 + b2 # Thêm + b2
    Y_hat = sigmoid(Z2)

    # --- Tính Loss (Giữ nguyên phong cách vector 1^T) ---
    L = -(Y * np.log(Y_hat + 1e-15) + (1 - Y) * np.log(1 - Y_hat + 1e-15))
    
    # Điểm nhấn: Nén vector L thành 1 số duy nhất bằng vector 1^T
    ones_n = np.ones((1, N)) # Sử dụng N thay vì 1000
    total_loss = (ones_n @ L) / N 
    E = total_loss[0, 0] 

    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {E:.4f}")

    # --- Backward ---
    dE_dZ2 = (1/N) * (Y_hat - Y)
    dW2 = H.T @ dE_dZ2
    db2 = np.sum(dE_dZ2, axis=0, keepdims=True) # Đạo hàm b2
    
    dE_dH = dE_dZ2 @ W2.T 
    # Ma trận Jacobian
    dH_dZ1 = H * (1 - H)
    dZ1_grad = dE_dH * dH_dZ1 # Biến trung gian để tính dW1 và db1

    dZ1_dW1 = X.T
    # Gradient cho W1
    dW1 = X.T @ dZ1_grad
    db1 = np.sum(dZ1_grad, axis=0, keepdims=True) # Đạo hàm b1

    # Update
    W1 -= lr * dW1
    b1 -= lr * db1 # Cập nhật b1
    W2 -= lr * dW2
    b2 -= lr * db2 # Cập nhật b2

# --- 3. Dự đoán ---
def predict(X_input, W1, b1, W2, b2): # Thêm b1, b2 vào tham số
    Z1_ts = X_input @ W1 + b1
    H_ts = sigmoid(Z1_ts) 
    Z2_ts = H_ts @ W2 + b2
    Y_ts = sigmoid(Z2_ts) 
    return Y_ts

# --- Đánh giá ---
res1 = predict(X1_test, W1, b1, W2, b2)
res2 = predict(X2_test, W1, b1, W2, b2)

acc1 = np.mean(res1 < 0.5) * 100
acc2 = np.mean(res2 >= 0.5) * 100

print("-" * 30)
print(f"Cụm 0 (Hẹp) Acc: {acc1:.2f}%")
print(f"Cụm 1 (Rộng) Acc: {acc2:.2f}%")
print("-" * 30)

# (Phần vẽ đồ thị plot_decision_boundary sẽ chạy mượt mà với hàm predict này)
