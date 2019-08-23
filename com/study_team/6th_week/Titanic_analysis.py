# -*- encoding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from texttable import Texttable


plt.style.use('mystyle')
data = pd.read_excel('./data/data.xlsx')
group = data.groupby('Embarked')


# 1. 按照港口分类，求出年龄、价格的统计量
age_stats = pd.DataFrame()
fare_stats = pd.DataFrame()

age_stats['mean'] = group.mean()['Age']
age_stats['var'] = group.var()['Age']
age_stats['std'] = group.std()['Age']
age_stats['cv'] = age_stats['std']/age_stats['mean']
age_stats['count'] = group.count()['Age']
age_stats['min'] = group.min()['Age']
age_stats['median'] = group.median()['Age']
age_stats['max'] = group.max()['Age']

fare_stats['mean'] = group.mean()['Fare']
fare_stats['var'] = group.var()['Fare']
fare_stats['std'] = group.std()['Fare']
fare_stats['cv'] = fare_stats['std']/fare_stats['mean']
fare_stats['count'] = group.count()['Fare']
fare_stats['min'] = group.min()['Fare']
fare_stats['median'] = group.median()['Fare']
fare_stats['max'] = group.max()['Fare']

age_stats = age_stats.apply(lambda x: round(x, 2))
fare_stats = fare_stats.apply(lambda x: round(x, 2))


age_header = [age_stats.index.name]
age_header.extend(age_stats.columns.get_values())
age_rows = np.c_[age_stats.index.values, age_stats.values]

age_tb = Texttable()
age_tb.set_cols_align(['c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c'])
age_tb.set_cols_dtype(['t', 't', 't', 't', 't', 'i', 't', 't', 't'])
age_tb.header(age_header)
age_tb.add_rows(age_rows, header=False)
print('=============================  Age Statistics  =============================\n', age_tb.draw(), '\n')

fare_header = [fare_stats.index.name]
fare_header.extend(fare_stats.columns.get_values())
fare_rows = np.c_[fare_stats.index.values, fare_stats.values]

fare_tb = Texttable()
fare_tb.set_cols_align(['c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c'])
fare_tb.set_cols_dtype(['t', 't', 't', 't', 't', 'i', 't', 't', 't'])
fare_tb.header(fare_header)
fare_tb.add_rows(fare_rows, header=False)
print('==============================  Fare Statistics  ==============================\n', fare_tb.draw(), '\n')


# 2. 画出价格的分布图像，验证数据服从何种分布（正态、卡方、T）？
fare = data['Fare'].copy().values
fare.sort()

# 2.1 验证正态分布
W, p = stats.shapiro(fare)
print("W=%.4f, p=%.4f" % (W, p))
'''
经过Shapiro-Wilk检验得出：
alpha=0.05
p = 0，p < alpha，拒绝原假设
W=0.5257，W的值越接近1就越表明数据和正态分布拟合得越好
所以价格数据不符合正态分布
'''
D, p = stats.kstest(fare, 'norm')
print("D=%.4f, p=%.4f" % (D, p))
'''
p = 0，p < alpha，拒绝原假设，价格数据不符合正态分布
'''


loc, scale = stats.norm.fit(fare)
y = stats.norm.pdf(fare, loc, scale)
plt.hist(fare, bins=10, density=True, label='Fare Distribution')
plt.plot(fare, y, color='#E78A61', label='normal fit')
plt.xlabel("Fare")
plt.ylabel("Density")
plt.title('Fare Distribution')
plt.legend()
plt.show()

x2 = stats.norm.rvs(loc=loc, scale=scale, size=len(fare))
D, p = stats.ks_2samp(fare, x2)
print("D=%.4f, p=%.4f" % (D, p))
'''
p = 0， p < alpha，拒绝原假设，价格数据不符合正态分布
'''

# 2.2 验证卡方分布
df, loc, scale = stats.chi2.fit(fare)
y = stats.chi2.pdf(fare, df, loc, scale)
plt.hist(fare, bins=10, density=True, label='Fare Distribution')
plt.plot(fare, y, color='#E78A61', label='chi2 fit')
plt.xlabel("Fare")
plt.ylabel("Density")
plt.title('Fare Distribution')
plt.legend()
plt.show()

x2 = stats.chi2.rvs(df=df, loc=loc, scale=scale, size=len(fare))
D, p = stats.ks_2samp(fare, x2)
print("D=%.4f, p=%.4f" % (D, p))
'''
p = 0，p < alpha，拒绝原假设，价格数据不符合卡方分布
'''


# 2.3 验证t分布
df, loc, scale = stats.t.fit(fare)
y = stats.t.pdf(fare, df, loc, scale)
plt.hist(fare, bins=10, density=True, label='Fare Distribution')
plt.plot(fare, y, color='#E78A61', label='t fit')
plt.xlabel("Fare")
plt.ylabel("Density")
plt.title('Fare Distribution')
plt.legend()
plt.show()

x2 = stats.t.rvs(df=df, loc=loc, scale=scale, size=len(fare))
D, p = stats.ks_2samp(fare, x2)
print("D=%.4f, p=%.4f" % (D, p))
'''
p = 0，p < alpha，拒绝原假设，价格数据不符合t分布
'''


# 3. 按照港口分类，验证S与Q两个港口间的价格之差是否服从某种分布
s_fare = data[data['Embarked'] == 'S']['Fare'].copy().values
q_fare = data[data['Embarked'] == 'Q']['Fare'].copy().values
c_fare = data[data['Embarked'] == 'C']['Fare'].copy().values

'''
设
E(X1)是独立的抽自S港的价格总体的一个容量为n1的样本的均值，n1<=554
E(X2)是独立的抽自Q港的价格总体的一个容量为n2的样本的均值，n1<=28
E(X3)是独立的抽自C港的价格总体的一个容量为n3的样本的均值，n1<=130
因为总体不服从正态分布，所以需要当n比较大时，一般要求n>=30，两个样本均值之差的抽样分布可近似为正态分布
因为X2的总体容量为28，其样本容量不可能超过30，故其S港和Q港两个样本均值之差（E(X1)-E(X2)）的抽样分布不服从正态分布
因为X3的总体容量为130，故其S港和C港两个样本均值之差（E(X1)-E(X3)）的抽样分布近似服从正态分布，其均值和方差分别为
E(E(X1) - E(X3)) = E(E(X1)) - E(E(X3)) = μ1 - μ3
D(E(X1) + E(X3)) = D(E(X1)) + D(E(X3)) = σ1²/n1 + σ3²/n3 
'''


mu = np.mean(s_fare) - np.mean(c_fare)
sigma = np.sqrt(np.var(s_fare, ddof=1)/len(s_fare) + np.var(c_fare, ddof=1)/len(c_fare))

x = np.arange(- 100, 20)
y = stats.norm.pdf(x, mu, sigma)
plt.plot(x, y)
plt.xlabel("S_Fare - C_Fare")
plt.ylabel("Density")
plt.title('Fare difference between S and C\n$\mu=%.2f,\quad \sigma=%.2f$' % (mu, sigma))
plt.show()
