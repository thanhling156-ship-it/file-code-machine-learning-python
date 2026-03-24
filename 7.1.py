import matplotlib.pyplot as plt
import numpy as np

# --- 1. Chuẩn bị dữ liệu ---
n_samples_train = 100
n_samples_test = 40
center = np.array([0, 0])
sigma1 = 0.2  # Cụm hẹp (Nhãn 0)
sigma2 = 0.6  # Cụm rộng (Nhãn 1)

# Tạo dữ liệu train
X1 = np.random.randn(n_samples_train, 2) * sigma1 + center
X2 = np.random.randn(n_samples_train, 2) * sigma2 + center
X1_test = np.random.randn(n_samples_test, 2) * sigma1 + center
X2_test = np.random.randn(n_samples_test, 2) * sigma2 + center

# Gộp X1 và X2 lại thành một "đội quân" duy nhất (200, 2)
X = np.vstack((X1, X2))

# Tạo nhãn tương ứng: 100 số 0 đầu tiên, 100 số 1 tiếp theo
y = np.vstack((np.zeros((n_samples_train, 1)), np.ones((n_samples_train, 1))))


# --- 2. Khởi tạo Trọng số ---
W1 = np.random.randn(2, 100) * 0.1
W2 = np.random.randn(100, 1) * 0.1
lr = 0.05 # Tăng nhẹ lr vì có ReLU

for i in range(1001):
    #Forward
    Z1 = X @ W1
    H = np.maximum(0,Z1)

    Y = H @ W2
    n = y.shape[0]
    E = np.mean((Y-y)**2) # trừ rồi tính Trung bình

    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {E:.4f}")

    #Backward
    #dE = []
    #for i in range(n):
    #dE.append((2/n) * (Y[i] - y[i]))
    #đạo hàm theo từng biến Yn nên Yn-1 sẽ là const => chỉ còn biến Yn
    dE = (2/n)*(Y-y)

    dW2 = H.T @ dE
    
    #---tinh dW1---
    dY = W2.T
    dH = dE @ dY
    dH[Z1 <= 0] = 0
    #ReLU' là 1 mask, chỉ có nghĩa khi áp lên matrix khác, vậy nên phải tính dH
    #vì H là kết quả, mà có thể sau này dùng activation khác, nên lấy gốc vẫn tốt hơn
    #Vậy nên, về mặt logic, lệnh dH[Z1 <= 0] = 0 và dH[H <= 0] = 0 cho ra kết quả y hệt nhau đối với ReLU.
    
    dZ1 = X.T

    #nhân input ở ngoài cùng của dWi
    dW1 = dZ1 @ dH


    W1 -= lr * dW1
    W2 -= lr * dW2

def predict(X_input, W1, W2):
    # Quy trình Forward y hệt lúc Train
    Z1_ts = X_input @ W1
    H_ts = np.maximum(0, Z1_ts) # CỰC KỲ QUAN TRỌNG: Phải có ReLU ở đây
    Y_ts = H_ts @ W2
    return Y_ts

# Dự đoán cho từng nhóm
res1 = predict(X1_test, W1, W2)
res2 = predict(X2_test, W1, W2)

# Tính độ chính xác (Giả sử ngưỡng phân loại là 0.5)
# Vì nhãn là 0 và 1, nên kết quả gần 0 là đúng cho cụm 1, gần 1 là đúng cho cụm 2
acc1 = np.mean(res1 < 0.5) * 100
acc2 = np.mean(res2 >= 0.5) * 100

print("-" * 30)
print(f"{'NHÓM ĐIỂM':<15} | {'DỰ ĐOÁN TB':<12} | {'ĐỘ CHÍNH XÁC'}")
print("-" * 30)
print(f"{'Cụm 0 (Hẹp)':<15} | {np.mean(res1):<12.4f} | {acc1:.2f}%")
print(f"{'Cụm 1 (Rộng)':<15} | {np.mean(res2):<12.4f} | {acc2:.2f}%")
print("-" * 30)
print(f"Tổng độ chính xác: {(acc1 + acc2) / 2:.2f}%")

# --- 4. VẼ ĐỒ THỊ QUAN SÁT ---
def plot_decision_boundary(X, y, W1, W2):
    # Tạo một lưới điểm (Grid) bao phủ vùng dữ liệu
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Dự đoán cho toàn bộ lưới điểm
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z_grid = predict(grid_points, W1, W2)
    Z_grid = Z_grid.reshape(xx.shape)

    # Vẽ vùng dự đoán (Bản đồ nhiệt)
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z_grid, cmap=plt.cm.RdBu, alpha=0.6)
    plt.colorbar(label="Xác suất/Giá trị dự đoán")

    # Vẽ các điểm dữ liệu thực tế
    plt.scatter(X1[:, 0], X1[:, 1], color='red', edgecolors='k', label='Cụm 0 (Hẹp)')
    plt.scatter(X2[:, 0], X2[:, 1], color='blue', edgecolors='k', label='Cụm 1 (Rộng)')
    
    plt.title("Đường biên quyết định của Neural Network tự chế")
    plt.xlabel("Tọa độ X")
    plt.ylabel("Tọa độ Y")
    plt.legend()
    plt.show()

# Gọi hàm vẽ
plot_decision_boundary(X, y, W1, W2)
