import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import shap

# 定义函数以计算描述符
def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # 计算描述符
        return [
            Chem.rdMolDescriptors.CalcAMR(mol),  # 摩尔折射率
            Chem.rdMolDescriptors.CalcGrossFormula(mol),  # 经验式
            Descriptors.MolWt(mol),  # 分子量
            Descriptors.MolLogP(mol),  # 辛醇/水分配系数的对数值
            Descriptors.TPSA(mol),  # 极性表面积
            Descriptors.LabuteASA(mol),  # Labute 极性表面积
            Descriptors.RingCount(mol),  # 环的数量
            Chem.rdMolDescriptors.CalcNumRotatableBonds(mol),  # 可旋转键的数量
            Chem.rdMolDescriptors.CalcNumHBA(mol),  # 氢键供体数量
            Chem.rdMolDescriptors.CalcNumHBD(mol),  # 氢键受体数量
            Descriptors.ExactMolWt(mol),  # 精确分子量
            Chem.AllChem.MMFFGetMolPartialCharges(mol)[0] if mol.GetNumConformers() > 0 else 0,  # MMFF94 方法计算的部分电荷
            Chem.rdMolDescriptors.CalcLabuteASA(mol, includeHydrogen=True),  # 考虑氢原子的 Labute 极性表面积
            Chem.rdMolDescriptors.CalcTPSA(mol),  # 极性表面积
            Chem.rdMolDescriptors.CalcFractionCSP3(mol),  # CSP3 杂化原子的分数
        ]
    else:
        # 如果分子无法从Smiles字符串创建，返回固定长度的零值列表
        return [0] * 12

# 读取数据
data = pd.read_csv('Smiles.csv')
smiles = data['structure']  # 假设Smiles字符串存储在'structure'列
labels = data['label']  # 假设标签存储在'label'列

# 应用函数计算描述符，并确保每个描述符列表的长度相同
descriptors = smiles.apply(get_descriptors)

# 将描述符转换为DataFrame，然后转换为NumPy数组
descriptor_df = pd.DataFrame(descriptors.tolist())
X_descriptors = descriptor_df.values

# 生成分子指纹
def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)

fingerprints = smiles.apply(get_fingerprint)

# 将分子指纹转换为NumPy数组
X_fingerprints = np.array(list(fingerprints), dtype=object)

# 合并描述符和指纹特征
X_combined = np.hstack((X_descriptors, X_fingerprints))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_combined, labels, test_size=0.2, random_state=42)

# 定义 Random Forest 分类器和参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 5, 8, 15],
    'min_samples_split': [2, 5, 10]
}

# 使用 GridSearchCV
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Calculate SHAP values with disabled additivity check
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Plot SHAP values
shap.summary_plot(shap_values, X_test, feature_names=[f'bit_{i}' for i in range(X_test.shape[1])])

# Predictions for test and training sets
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

# Plotting
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, alpha=0.5, label='Train')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Training Data')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, alpha=0.5, color='red', label='Test')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Testing Data')
plt.legend()

plt.tight_layout()
plt.show()

print(f'Training Accuracy: {accuracy_train}')
print(f'Testing Accuracy: {accuracy_test}')
print(f'Training MSE: {mse_train}')
print(f'Testing MSE: {mse_test}')