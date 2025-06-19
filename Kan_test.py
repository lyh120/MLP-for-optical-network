import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from kan import KAN
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 生成模拟时间序列数据
def generate_synthetic_data(length=1000):
    x = np.linspace(0, 20*np.pi, length)
    main_wave = np.sin(x)
    seasonality = 0.5 * np.sin(4*x)
    noise = 0.1 * np.random.randn(length)
    y = main_wave + seasonality + noise + 2
    return pd.DataFrame({'load': y})

# 数据准备与预处理
data = generate_synthetic_data()
dataset = data.values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# 创建时间窗口数据集
def create_dataset(data, look_back=72, pred_steps=24):
    X, Y = [], []
    for i in range(len(data)-look_back-pred_steps+1):
        X.append(data[i:(i+look_back), 0])
        Y.append(data[(i+look_back):(i+look_back+pred_steps), 0])
    return np.array(X), np.array(Y)

look_back = 72
pred_steps = 24
X, Y = create_dataset(dataset, look_back, pred_steps)

# 划分数据集（训练60%，验证20%，测试20%）
total_samples = len(X)
train_size = int(total_samples * 0.6)
val_size = int(total_samples * 0.2)
test_size = total_samples - train_size - val_size

X_train, Y_train = X[:train_size], Y[:train_size]
X_val, Y_val = X[train_size:train_size+val_size], Y[train_size:train_size+val_size]
X_test, Y_test = X[train_size+val_size:], Y[train_size+val_size:]

# 转换为PyTorch张量
X_train = torch.tensor(X_train).unsqueeze(-1).float()
Y_train = torch.tensor(Y_train).float()
X_val = torch.tensor(X_val).unsqueeze(-1).float()
Y_val = torch.tensor(Y_val).float()
X_test = torch.tensor(X_test).unsqueeze(-1).float()

# 定义改进的LSTM-KAN模型
class EnhancedLSTMKAN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, output_dim=24):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, 
                          batch_first=True, 
                          num_layers=2,
                          dropout=0.2)
        
        # 使用标准KAN参数
        self.kan = KAN(width=[hidden_dim, 256, output_dim],
                     grid=5,
                     k=3)
        
        # 在KAN后添加自定义激活层
        self.activation = nn.SiLU()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        kan_out = self.kan(last_step)
        return self.activation(kan_out)  # 应用自定义激活

# 模型初始化与训练配置
model = EnhancedLSTMKAN()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.HuberLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

# 训练循环（含验证和早停）
best_val_loss = float('inf')
patience_counter = 0
patience = 10
train_history = []
val_history = []

for epoch in range(10):
    # 训练阶段
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = loss_fn(outputs, Y_train)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    train_loss = loss.item()
    
    # 验证阶段
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = loss_fn(val_outputs, Y_val).item()
    
    # 学习率调度
    scheduler.step(val_loss)
    
    # 记录训练过程
    train_history.append(train_loss)
    val_history.append(val_loss)
    
    # 打印详细训练信息
    print(f"Epoch {epoch+1:03d} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    # 早停机制
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

# 加载最佳模型进行测试
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = loss_fn(test_outputs, torch.tensor(Y_test).float()).item()

print(f"\nFinal Test Loss: {test_loss:.4f}")

# 结果可视化
def inverse_transform(data):
    return scaler.inverse_transform(data.reshape(-1, 1)).flatten()

# 绘制损失曲线
plt.figure(figsize=(12, 6))
plt.plot(train_history, label='Training Loss')
plt.plot(val_history, label='Validation Loss')
plt.title("Training Progress")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# 绘制预测对比（取前5个测试样本）
plt.figure(figsize=(15, 10))
for i in range(5):
    plt.subplot(5, 1, i+1)
    true = inverse_transform(Y_test[i])
    pred = inverse_transform(test_outputs[i].numpy())
    plt.plot(true, label='True', marker='o')
    plt.plot(pred, label='Predicted', linestyle='--', marker='x')
    plt.title(f"Test Sample {i+1}")
    plt.ylabel("Value")
    plt.legend()
plt.tight_layout()
plt.show()