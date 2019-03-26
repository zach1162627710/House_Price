import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
#matplotlib inline

data_train = pd.read_csv('D:/house-prices-advanced-regression-techniques/train.csv')
# print(data_train.head(5))
#print(data_train['SalePrice'].describe())
# sns.distplot(data_train['SalePrice'])  #画直方图
# plt.show()

# CentralAir
# var = 'CentralAir'
# data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
# #数据合并，“SalePrice”和“CentralAir”列合并到data
# fig = sns.boxplot(x=var, y="SalePrice", palette="Set2",data=data)
# fig.axis(ymin=0, ymax=800000);
#plt.show()

# data = pd.concat([data_train['SalePrice'],data_train['OverallQual']],axis = 1)
# fig = sns.boxplot(x= "OverallQual", y="SalePrice",palette = "Set2",data = data)
# fig.axis(ymin=0, ymax=800000);
#plt.show()

#data = pd.concat([data_train['YearBuilt'],data_train['SalePrice']],axis=1)
#data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
# plt.subplots(figsize=(40, 12))  #画板的规模，长 40，高 12
# fig = sns.boxplot(x="YearBuilt", y="SalePrice", data=data)
# fig.axis(ymin=0, ymax=800000);
#plt.show()


# data = pd.concat([data_train["SalePrice"],data_train["YearBuilt"]],axis=1)
# data.plot.scatter(x="YearBuilt",y="SalePrice",ylim=(0,800000))


# data = pd.concat([data_train['SalePrice'],data_train['GrLivArea']],axis=1)
# data.plot.scatter(x='GrLivArea',y='SalePrice',ylim=(0,800000))
# plt.show()


# var=['GarageArea','GarageCars']
# for index in range(2):
#     data = pd.concat([data_train['SalePrice'],data_train[var[index]]],axis=1)
#     data.plot.scatter(x=var[index],y='SalePrice',ylim=(0,800000))


f_name=['CentralAir','Neighborhood']
for i in f_name:
    label = preprocessing.LabelEncoder()
    print(label)
    data_train[i] = label.fit_transform(data_train[i])

corrmat = data_train.corr()  ##相关系数矩阵，即任意2类属性之间的关系数  而组成的矩阵
# f, ax = plt.subplots(figsize=(9, 9))
# sns.heatmap(corrmat, vmax=1, square=True)#画出热图的意思
# plt.show()

k = 10     # 关系矩阵中将显示10个特征  ['SalePrice'].index
df = corrmat.nlargest(k, 'SalePrice')
cols = df['SalePrice'].index
# print(cols)
# print(data_train[cols].values)    #cols：根据SalePrice得出的SalePrice最大10位的参数
cm = np.corrcoef(data_train[cols].values.T)   #根据传进去的10计算相关系数矩阵
print(cm)
sns.set(font_scale=1.25)                    #设置字体
hm = sns.heatmap(cm, cbar=True, annot=True, \
                 square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


sns.set()
cols = ['SalePrice','OverallQual','GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
sns.pairplot(data_train[cols], size = 2.5)
plt.show()