import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. 匯入資料並正確設定標題
df = pd.read_csv("boston_house_prices.csv", header=1)  # header=1 讓資料的第二行作為欄位名稱

# 檢查資料結構，確認標題行正確
print(df.head())

# 2. 檢查是否有缺失值
print("資料中缺失值的數量：")
print(df.isnull().sum())

# 3. 列出房價的統計數據
print(f"最高房價: {df['MEDV'].max()}")
print(f"最低房價: {df['MEDV'].min()}")
print(f"平均房價: {df['MEDV'].mean()}")
print(f"中位數房價: {df['MEDV'].median()}")

# 4. 房價分布直方圖 (以10為區間)
plt.figure(figsize=(8,6))
plt.hist(df['MEDV'], bins=range(0, 51, 10), edgecolor='black')
plt.title('房價分布')
plt.xlabel('房價 ($1000)')
plt.ylabel('頻率')
plt.show()

# 5. RM 值四捨五入並分析不同 RM 值的平均房價
df['RM_rounded'] = df['RM'].round()

# 按 RM 值分組並計算每組的平均房價
grouped_rm = df.groupby('RM_rounded')['MEDV'].mean()
print("不同RM值下的平均房價：")
print(grouped_rm)

# 6. 繪製 RM 值的直方圖
plt.figure(figsize=(8,6))
plt.hist(df['RM_rounded'], bins=np.arange(df['RM_rounded'].min(), df['RM_rounded'].max() + 1, 1), edgecolor='black')
plt.title('不同 RM 值的分布')
plt.xlabel('RM值（四捨五入）')
plt.ylabel('頻率')
plt.show()

# 7. 使用線性回歸預測房價
# 特徵選擇：使用 "RM" 來預測房價
X = df[['RM']]  # 特徵
y = df['MEDV']  # 目標變數

# 分割資料集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立線性回歸模型
model = LinearRegression()
model.fit(X_train, y_train)

# 預測房價
y_pred = model.predict(X_test)

# 評估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"均方誤差 (MSE): {mse}")
print(f"R^2: {r2}")

# 8. 繪製預測結果與實際結果的比較圖
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('預測房價 vs 實際房價')
plt.xlabel('實際房價')
plt.ylabel('預測房價')
plt.show()
