# -*- encoding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

plt.style.use('mystyle')
data = pd.read_csv('http://jse.amstat.org/datasets/normtemp.dat.txt',
                   header=None, sep='\s+', names=['temperature', 'gender', 'heart_rate'])


# 1. Is the true population mean really 98.6 degrees F?
mu = 98.6
x = data['temperature'].copy().values
n = len(x)
mean = np.mean(x)
std = np.std(x, ddof=1)
t, p = stats.ttest_1samp(x, mu)
print("t=%.2f, p=%.7f" % (t, p))
interval = stats.t.interval(0.05, n - 1, mean, std)
print("置信区间为： (%.4f, %.4f)" % interval)
'''
经过单一样本t检验得出：t=-5.45, p=0.0000002
因为 p=0.0000002, p > 0.05 所以拒绝原假设
计算 α=0.05 时的置信区间为 interval = (98.2032, 98.2953)，而 98.6 不在此区间，所以拒绝原假设
但是总体均值98.6不一定是错的，有可能是选取到了特殊样本
'''


# 2. Is the distribution of temperatures normal?
x.sort()
'''
normaltest正态性检验测试样本是否与正态分布不同。
此函数测试样本来自正态分布的零假设。 
它基于D'Agostino和Pearson的测试，结合了倾斜和峰度，以产生正常的综合测试。
'''
k2, p = stats.normaltest(x)
alpha = 1e-3
# 零假设（null hypothesis），统计学术语，又称原假设，指进行统计检验时预先建立的假设。 零假设成立时，有关统计量应服从已知的某种概率分布。
if p < alpha:
    print("The null hypothesis can be rejected")
else:
    print("The null hypothesis cannot be rejected")
print("k2=%.4f, p=%.4f" % (k2, p))
'''
经过正态性检验得出：
p = 0.2587，p > alpha， 所以接受原假设，认为体温是符合正态分布的
'''

'''
执行Shapiro-Wilk检验以获得正态性。
Shapiro-Wilk检验测试零假设，即数据是否来自正态分布。
'''
W, p = stats.shapiro(x)
print("W=%.4f, p=%.4f" % (W, p))
'''
经过Shapiro-Wilk检验得出：
p = 0.2332，p > alpha， 所以接受原假设，认为体温是符合正态分布的
'''

# 做出正态分布图
loc, scale = stats.norm.fit(x)
plt.hist(x, bins=10, normed=True, label='sample distribution')
plt.plot(x, stats.norm.pdf(x, loc, scale), label='normal fit')
plt.xlabel('temperature $(F)$')
plt.ylabel('sample size')
plt.title('Temperature Distribution: $E(X)=%.2f, \quad SD(X)=%.2f$' % (np.mean(x), np.std(x, ddof=1)))
plt.xlim(96, 101)
plt.legend()
plt.show()

# 3. At what temperature should we consider someone's temperature to be "abnormal"?
interval = stats.t.interval(0.95, n - 1, mean, std)
print("置信区间为： (%.4f, %.4f)" % interval)
'''
interval = (96.7986, 99.6999) 所以体温小于 96.7986 或者 大于 99.6999 是异常的
'''

# 4. Is there a significant difference between males and females in normal temperature?
x1 = data[data['gender'] == 1]['temperature'].copy().values
x2 = data[data['gender'] == 2]['temperature'].copy().values
# 用levene test检验两组数据的方差齐性，若p>0.05则认为方差是相等的。
statistic, pvalue = stats.levene(x1, x2)
# 如果方差不齐性，则equal_var=False
t, p = stats.ttest_ind(x1, x2, equal_var=True)
print("t=%.3f，P=%.3f" % (t, p))
'''
经过两独立样本t检验计算出：
t=-2.285，P=0.024
在α=0.05的检验水准上，拒绝原假设，但也有可能是样本选取有偏差，女性和男性体温是否真正有差异还需进一步讨论。
'''
# 分别作出女性和男性的分布图
loc1, scale1 = stats.norm.fit(x1)
plt.plot(x1, stats.norm.pdf(x1, loc1, scale1), label='x1')
loc2, scale2 = stats.norm.fit(x2)
plt.plot(x2, stats.norm.pdf(x2, loc2, scale2), label='x2')
plt.xlabel('temperature $(F)$')
plt.ylabel('sample size')
plt.title('Temperature Distribution: \n$E(X_1)=%.2f, \quad SD(X_1)=%.2f$, $E(X_2)=%.2f, \quad SD(X_2)=%.2f$' % (np.mean(x1), np.std(x1, ddof=1), np.mean(x2), np.std(x2, ddof=1)))
plt.xlim(96, 101)
plt.legend()
plt.show()

# 5. Is there a relationship between body temperature and heart rate?
y = data['heart_rate'].copy().values
r, p = stats.pearsonr(x, y)
print("r=%.2f, p=%.3f" % (r, p))
'''
皮尔森相关系数与其他相关系数一样，在-1和+1之间变化，0表示没有相关性。
-1或+1的相关系数意味呈完全的线性关系。 正相关意味着随着x的增加，y也增加。 负相关意味着随着x增加，y减小。
计算得到皮尔森相关系数 r = 0.25， 说明体温与心率相关性并不高。
p值粗略地表示了不相关的总体系统产生具有Pearson相关性的样本的概率。
计算得到 p = 0.004
'''

corr, p = stats.spearmanr(x, y)
print("corr=%.2f, p=%.3f" % (corr, p))
'''
Spearman相关系数同理， 计算得到： corr = 0.28， 仍然说明体温与心率相关性并不高。
p值并不完全可靠，但对于大于500左右的数据集可能是合理的。
计算得到 p = 0.001
'''

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print("slope: %f    intercept: %f" % (slope, intercept))
print("R-squared: %f" % r_value**2)
plt.plot(x, y, 'o', label='original data')
plt.plot(x, intercept + slope*x, label='fitted line')
plt.xlabel('$x$: temperature $(F)$')
plt.ylabel('$y$: heart rate $(n/min)$')
plt.title('Temperature & Heart rate liner regression: \n$y=%.2fx%.2f$,  $R^2=%.4f$' % (slope, intercept, r_value**2))
plt.legend()
plt.show()
'''
R-squared是线性回归的决定系数（coefficient ofdetermination），有的翻译为判定系数，也称为拟合优度。
决定系数反应了y的波动有多少百分比能被x的波动所描述，即表征因变量y的变异中有多少百分比,可由自变量x来解释。
表达式：R2=SSR/SST=1-SSE/SST
其中：SST=SSR+SSE，SST(total sum of squares)为总平方和，
SSR(regression sum of squares)为回归平方和，SSE(error sum of squares) 为残差平方和。

计算得到： R² = 0.0643， 仍然说明体温与心率相关性并不高。
'''

# 6. Were the original temperatures taken on a Centigrade or Fahrenheit scale?
data['temperature_centigrade'] = data['temperature'].apply(lambda x: (x - 32) * 5/9)
x = data['temperature_centigrade'].copy().values
plt.hist(x, bins=10, normed=True, label='sample distribution')
plt.xlabel('temperature $(F)$')
plt.ylabel('sample size')
plt.title('Temperature Distribution: $E(X)=%.2f, \quad SD(X)=%.2f$' % (np.mean(x), np.std(x, ddof=1)))
plt.xlim(35, 40)
plt.legend()
plt.show()
