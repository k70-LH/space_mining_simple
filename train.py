import joblib
import os
import numpy as np
from sklearn.naive_bayes import GaussianNB
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, size=100)

# 构建并训练轻量模型
model = GaussianNB()
model.fit(X, y)

# 创建模型保存文件夹
os.makedirs("model", exist_ok=True)
# 保存训练好的模型
joblib.dump(model, "model/simple_model.pkl")

# 生成简易评估文件
with open("eval_result.txt", "w", encoding="utf-8") as f:
    f.write("太空采矿极简ML模型\n")
    f.write("训练完成状态：成功\n")
    f.write("模型类型：朴素贝叶斯分类器\n")
    f.write("适配GitHub CI/CD自动部署")

print("✅ 模型训练完毕，已保存至model文件夹，评估文件生成完成！")