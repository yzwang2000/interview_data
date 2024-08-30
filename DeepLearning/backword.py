import numpy as np
import matplotlib.pyplot as plt

train_loss = [] # 记录每一次的损失，为了可视化
def train(X, Y, lr, num_epochs):
    # 随机生成权重参数
    w1 = np.random.randn(3, 2)
    w2 = np.random.randn(1, 3)

    for epoch in range(num_epochs):
        # 正向传播
        H1 = np.dot(w1, X)   # 第1个隐藏层
        y = np.dot(w2, H1)   # 输出层
        loss = np.sum((Y - y) ** 2) / 2
        print(f'{epoch+1}轮的损失：{loss}')
        train_loss.append(loss)

        # 求误差
        s2 = y - Y              # 求的是 Loss 对 y 的偏导
        s1 = np.dot(w2.T, s2)   # 求的是 Loss 对 z 的偏导

        # 求偏导
        dw1 = np.dot(s1, X.T)   # 求的是 Loss 对 w^1 的偏导
        dw2 = np.dot(s2, H1.T)  # 求的是 Loss 对 w^2 的偏导

        # 更新 w
        w1 = w1 - lr * dw1
        w2 = w2 - lr * dw2

    return w1, w2

# 创建数据
X = np.array([[40], [80]]) # 训练集, [2, 1]
Y = np.array([[60]])       # 真实数据, [1, 1]
# 训练
w1, w2 = train(X, Y, 0.0001, 10)
print(w1)
print(w2)

# 预测，相当于前向传播
H1 = np.dot(w1, X)
y = np.dot(w2, H1)
print(f'预测数据为{y}')
# 可视化
plt.plot(range(1,11), train_loss)
plt.title('train loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig("sine_wave.png")