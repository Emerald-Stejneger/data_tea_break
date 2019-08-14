# -*- encoding: utf-8 -*-
import numpy as np
from scipy import stats
from scipy.special import gamma
import matplotlib.pyplot as plt


plt.style.use('mystyle')


# 2 概率分布
# 2.1 基本概念
# 2.1.1 随机变量
'''
随机变量（random variable）表示随机试验各种结果的实值单值函数。随机事件不论与数量是否直接有关，都可以数量化，即都能用数量化的方式表达。
简单地说，随机变量是指随机事件的数量表现。某地若干名男性健康成人中，每人血红蛋白量的测定值等等。

另有一些现象并不直接表现为数量，例如人口的男女性别、试验结果的阳性或阴性等，但我们可以规定男性为1，女性为0，则非数量标志也可以用数量来表示。
这些例子中所提到的量，尽管它们的具体内容是各式各样的，但从数学观点来看，它们表现了同一种情况，
这就是每个变量都可以随机地取得不同的数值，而在进行试验或测量之前，我们要预言这个变量将取得某个确定的数值是不可能的。
按照随机变量可能取得的值，可以把它们分为两种基本类型：
离散型随机变量，见2.1.4
连续型随机变量，见2.1.5
'''

# 2.1.2 古典概率
'''
古典概率通常又叫事前概率，是指当随机事件中各种可能发生的结果及其出现的次数都可以由演绎或外推法得知，
而无需经过任何统计试验即可计算各种可能发生结果的概率。

传统概率的定义是由法国数学家拉普拉斯 ( Laplace ) 提出的。
如果一个随机试验所包含的单位事件是有限的，且每个单位事件发生的可能性均相等，则这个随机试验叫做拉普拉斯试验。
'''

# 2.1.3 条件概率
'''
条件概率（英语：conditional probability）就是事件A在事件B发生的条件下发生的概率。
条件概率表示为P（A|B），读作“A在B发生的条件下发生的概率”。

联合概率表示两个事件共同发生的概率。A与B的联合概率表示为 P(A ∩ B) 或者 P(A, B) 或者 P(AB) 。

需要注意的是，在这些定义中A与B之间不一定有因果或者时间序列关系。A可能会先于B发生，也可能相反，也可能二者同时发生。
A可能会导致B的发生，也可能相反，也可能二者之间根本就没有因果关系。

'''

# 2.1.4 离散变量
'''
离散型随机变量
即在一定区间内变量取值为有限个，或数值可以一一列举出来。例如某地区某年人口的出生数、死亡数，某药治疗某病病人的有效数、无效数等

离散变量指变量值可以按一定顺序一一列举，通常以整数位取值的变量。如职工人数、工厂数、机器台数等。
有些性质上属于连续变量的现象也按整数取值，即可以把它们当做离散变量来看待。
例如年龄、评定成绩等虽属连续变量，但一般按整数计算，按离散变量来处理。离散变量的数值用计数的方法取得。
'''

# 2.1.5 连续变量
'''
连续型随机变量
即在一定区间内变量取值有无限个，或数值无法一一列举出来。例如某地区男性健康成人的身长值、体重值，一批传染性肝炎患者的血清转氨酶测定值等。

在一定区间内可以任意取值的变量叫连续变量，其数值是连续不断的，相邻两个数值可作无限分割，即可取无限个数值。
'''

# 2.1.6 期望值
'''
一个离散性随机变量的期望值（或数学期望，亦简称期望，物理学中称为期待值）是试验中每次可能的结果乘以其结果概率的总和。
换句话说，期望值像是随机试验在同样的机会下重复多次，所有那些可能状态平均的结果，便基本上等同“期望值”所期望的数。
期望值可能与每一个结果都不相等。换句话说，期望值是该变量输出值的加权平均。
期望值并不一定包含于其分布值域，也并不一定等于值域平均值。
例如，掷一枚公平的六面骰子，其每次“点数”的期望值是3.5，计算如下：
E(X) = 1 * 1/6 + 2 * 1/6 + 3 * 1/6 + 4 * 1/6 + 5 * 1/6 + 6 * 1/6 = 3.5
不过如上所说明的，3.5虽是“点数”的期望值，但却不属于可能结果中的任一个，没有可能掷出此点数。
'''

# 2.2 离散变量概率分布
# 2.2.1 二项分布
'''
二项分布（英语：Binomial distribution）是n个独立的是/非试验中成功的次数的离散概率分布，其中每次试验的成功概率为p。
这样的单次成功/失败试验又称为伯努利试验。实际上，当n = 1时，二项分布就是伯努利分布。二项分布是显著性差异的二项试验的基础。
'''

n = 10
p = 0.3
k = np.arange(0, 21)
binomial = stats.binom.pmf(k, n, p)
plt.figure()
plt.plot(k, binomial, 'o-')
plt.title('Binomial: $P(X=k)=\left(\\frac{n!}{k!(n-k)!}\\right)p^k(1-p)^{n-k}, \quad (n=%i, p=%.1f)$' % (n, p))
plt.xlabel('Number of success')
plt.ylabel('Probability of success')
plt.xlim(0, 20)
plt.ylim(0)
plt.show()

binom_sim = stats.binom.rvs(n=10, p=0.3, size=10000)
plt.hist(binom_sim, bins=10, normed=True)
plt.xlabel('$x$')
plt.ylabel('density')
plt.title('Simulating Binomial Random Variables:  \n$E(X)=%.2f, \quad SD(X)=%.2f$' % (np.mean(binom_sim), np.std(binom_sim, ddof=1)))
plt.xlim(0, 10)
plt.show()


# 2.2.2 伯努利分布
'''
伯努利分布（英语：Bernoulli distribution，又名两点分布或者0-1分布。
若伯努利试验成功，则伯努利随机变量取值为1。若伯努利试验失败，则伯努利随机变量取值为0。
记其成功概率为 p (0 <= p <= 1)，失败概率为 q = 1 - p。
其期望值为： E(X) = p， 方差为 D(X) = pq
伯努利分布是二项分布在n = 1时的特殊情况。X ~ B(1, p)与X ~ Bern(p)的意思是相同的。
相反，任何二项分布B(n,p)都是n次独立伯努利试验的和，每次试验成功的概率为p。
'''

p = 0.3
k = np.arange(0, 2)
bernoulli = stats.bernoulli.pmf(k, p)
plt.figure()
plt.plot(k, bernoulli, 'o-')
plt.title('Bernoulli: $P(X=k)=p^k(1-p)^{1-k}, \quad (p=%.1f)$' % p)
plt.xlabel('Number of success')
plt.ylabel('Probability of success')
plt.xticks(np.arange(0, 2, 1))
plt.xlim(0, 1)
plt.ylim(0)
plt.show()

bernoulli_sim = stats.bernoulli.rvs(p=0.3, size=10000)
plt.hist(bernoulli_sim, bins=10, normed=True)
plt.xlabel('$x$')
plt.title('Simulating Bernoulli Random Variables:  \n$E(X)=%.2f, \quad SD(X)=%.2f$' % (np.mean(bernoulli_sim), np.std(bernoulli_sim, ddof=1)))
plt.xticks(np.arange(0, 2, 1))
plt.xlim(0, 1)
plt.show()


# 2.2.3 泊松分布
'''
二项分布是泊松二项分布的一个特殊情况。泊松二项分布是n次独立、不相同的伯努利试验(pi)的和。
如果X服从泊松二项分布，且p1 = … = pn =p，那么X ~ B(n, p)。
'''

rate = 2
n = np.arange(0, 10)
y = stats.poisson.pmf(n, rate)
plt.figure()
plt.plot(n, y, 'o-')
plt.title('Poisson: $P(X=k)=\\frac{\lambda^ke^{-\lambda}}{k!}, \quad (\lambda=%i)$' % rate)
plt.xlabel('Number of accident')
plt.ylabel('Probability of number of accident')
plt.xticks(np.arange(0, 10, 1))
plt.xlim(0, 9)
plt.ylim(0)
plt.show()

data = stats.poisson.rvs(mu=2, loc=0, size=1000)
plt.figure()
plt.hist(data, bins=9, normed=True)
plt.xlim(0, 10)
plt.xlabel('Number of accident')
plt.title('Simulating Poisson Random Variables:  \n$E(X)=%.2f, \quad D(X)=%.2f$' % (np.mean(data), np.std(data, ddof=1)))
plt.show()


# 2.3 连续变量概率分布
# 2.3.1 均匀分布
'''
连续型均匀分布，如果连续型随机变量 X 具有如下的概率密度函数，则称 X 服从 [a,b] 上的均匀分布（uniform distribution）,
记作 X ~ U[a, b]。期望值为： E(X) = (a + b)/2，方差为 D(X) = (b - a)²/12
'''

a, b = 2, 4
x = np.arange(0, 6, 0.01)
y = stats.uniform.pdf(x, a, b - a)
plt.figure()
plt.plot(x, y)
plt.title('Uniform: $f(x;a, b)= \\frac{1}{b-a}, \ a\leq x\leq b $\n$(a=%i, b=%i) \quad f(x;a, b)= 0, \quad x<a \ or \ x\geq b$' % (a, b))
plt.xlabel('$x$')
plt.ylabel('Probability density')
plt.xlim(0)
plt.ylim(0)
plt.show()


# 2.3.2 正态分布
'''
正态分布（英语：normal distribution）又名高斯分布（英语：Gaussian distribution），是一个非常常见的连续概率分布。
正态分布在统计学上十分重要，经常用在自然和社会科学来代表一个不明的随机变量。
若随机变量 X 服从一个位置参数为 μ 的正态分布，记为：
X ~ N(μ, σ²)
正态分布的数学期望值或期望值 μ 等于位置参数，决定了分布的位置；其方差 σ² 的开平方或标准差 σ 等于尺度参数，决定了分布的幅度。
正态分布的概率密度函数曲线呈钟形，因此人们又经常称之为钟形曲线（类似于寺庙里的大钟，因此得名）。
我们通常所说的标准正态分布是位置参数 μ = 0，尺度参数 σ² = 1的正态分布。
'''

mu = 0
sigma = 1
x = np.arange(-5, 5, 0.1)
y = stats.norm.pdf(x, mu, sigma)
plt.plot(x, y)
plt.title('Normal: $f(x;\mu,\sigma)=\\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\\frac{(x-\mu)^2}{2\sigma^2}},\quad (\mu=%.1f,\quad \sigma^2=%.1f$)' % (mu, sigma))
plt.xlabel('$x$')
plt.ylabel('Probability density')
plt.ylim(0)
plt.show()


# 2.3.3 指数分布
'''
指数分布（英语：Exponential distribution）是一种连续概率分布。指数分布可以用来表示独立随机事件发生的时间间隔，
比如旅客进入机场的时间间隔、打进客服中心电话的时间间隔、中文维基百科新条目出现的时间间隔等等。
指数分布的概率密度函数是：
x >= 0 时，f(x; λ) = λ * Exp(-λx)
x < 0 时，f(x; λ) = 0
其中λ > 0是分布的一个参数，常被称为率参数（rate parameter）。即每单位时间发生该事件的次数。
指数分布的区间是[0,∞)。 如果一个随机变量X 呈指数分布，则可以写作：X ~ Exp（λ）。
指数分布的均值为 E[X] = 1/λ ，方差为 D[X] = 1/(λ²)
'''

lambd = 0.5
x = np.arange(0, 15, 0.1)
y = lambd * np.exp(- lambd * x)
# y = stats.expon.pdf(x)
plt.plot(x, y)
plt.title('Exponential: $f(x;\lambda)=\lambda e^{-\lambda x}, \quad (\lambda=%.1f)$' % lambd)
plt.xlabel('$x$')
plt.ylabel('Probability density')
plt.xlim(0)
plt.ylim(0)
plt.show()

data = stats.expon.rvs(scale=2, size=1000)
plt.hist(data, bins=20, normed=True)
plt.xlim(0, 15)
plt.title('Simulating Exponential Random Variables:  \n$E(X)=%.2f,\quad D(X)=%.2f$' % (np.mean(data), np.std(data, ddof=1)))
plt.show()


# 2.3.4 Gamma分布
'''
伽玛分布（Gamma Distribution）是统计学的一种连续概率函数，是概率统计中一种非常重要的分布。又称Γ分布。
伽玛分布中的参数α，称为形状参数，β称为尺度参数。
令 X ~ Γ(α, β)， 且令 λ = 1/β （即 X ~ Γ(α, 1/λ) ），则指数分布的概率密度函数是：
f(x) = x^(α - 1) * λ^α * Exp(-λx)/Γ(α), (x > 0)
其中 Γ(α) 函数的特征为：
α ∈ Z+, Γ(α) = (α - 1)!
α ∈ R+, Γ(α) = (α - 1) * Γ(α - 1)
α = 1/2, Γ(α) = sqrt(π)

伽马分布的均值为 E[X] = α * β ，方差为 D[X] = α * β²

“指数分布”和“χ²分布”都是伽马分布的特例：
当形状参数 α = 1 时，伽马分布就是参数为 λ = 1/β 的指数分布，X ~ Exp（λ）
当 α = n/2，β = 1/2 时，伽马分布就是自由度为n的卡方分布，X²(n)
'''


def gamma_pdf(x, alpha, beta):
    if type(x) != list:
        return (x ** (alpha - 1)) * (beta ** alpha) * np.exp(- beta * x) / gamma(alpha)
    elif type(x) == list:
        return [np.power(i, (alpha - 1)) * np.power(beta, alpha) * np.exp(- beta * i) / gamma(alpha) for i in x]


alpha, beta = 1, 1/2
x = np.arange(0, 20, 0.1)
y = gamma_pdf(x, alpha, beta)
# y = stats.gamma.pdf(x, alpha)
plt.plot(x, y, label='$\\alpha$=%.1f, $\\beta$=%.1f' % (alpha, beta))
alpha, beta = 2, 1/2
y = gamma_pdf(x, alpha, beta)
plt.plot(x, y, label='$\\alpha$=%.1f, $\\beta$=%.1f' % (alpha, beta))
alpha, beta = 3, 1/2
y = gamma_pdf(x, alpha, beta)
plt.plot(x, y, label='$\\alpha$=%.1f, $\\beta$=%.1f' % (alpha, beta))
alpha, beta = 5, 1
y = gamma_pdf(x, alpha, beta)
plt.plot(x, y, label='$\\alpha$=%.1f, $\\beta$=%.1f' % (alpha, beta))
alpha, beta = 9, 2
y = gamma_pdf(x, alpha, beta)
plt.plot(x, y, label='$\\alpha$=%.1f, $\\beta$=%.1f' % (alpha, beta))
plt.title('Gamma: $f(x;\\alpha, \\beta)=\\frac{x^{\\alpha-1}\\beta^\\alpha e^{-\\beta x}}{\Gamma(\\alpha)}$')
plt.xlabel('$x$')
plt.ylabel('Probability density')
plt.xticks(np.arange(0, 20, 2))
plt.xlim(0, 20)
plt.ylim(0, 0.5)
plt.legend()
plt.show()


# 2.3.5 偏态分布
'''
偏态分布是与“正态分布”相对，分布曲线左右不对称的数据次数分布，是连续随机变量概率分布的一种。可以通过峰度和偏度的计算，衡量偏态的程度。
可分为正偏态和负偏态，前者曲线右侧偏长，左侧偏短；后者曲线左侧偏长，右侧偏短。
'''

a = 4
x = np.linspace(stats.skewnorm.ppf(0.01, a), stats.skewnorm.ppf(0.99, a), 100)
y = stats.skewnorm.pdf(x, a)
plt.plot(x, y)
plt.title('Skew Normal: $f(x;a)=2f(x;\mu,\sigma)F(ax;\mu,\sigma),\quad (a=%i$)' % a)
plt.xlabel('$x$')
plt.ylabel('Probability density')
plt.ylim(0)
plt.show()


# 2.3.6 Beta分布
'''
β分布是一个取值在 [0, 1] 之间的连续分布，它由两个形态参数α和β的取值所刻画。
β分布的形状取决于α和β的值。贝叶斯分析中大量使用了β分布。
'''

a, b = 0.5, 0.5
x = np.arange(0.01, 1, 0.01)
y = stats.beta.pdf(x, a, b)
plt.plot(x, y)
plt.title('Beta: $f(x;\\alpha,\\beta)=\\frac{\Gamma(\\alpha+\\beta)}{\Gamma(\\alpha)\Gamma(\\beta)}x^{\\alpha-1}(1-x)^{\\beta-1},\quad (\\alpha=%.1f, \\beta=%.1f)$' % (a, b))
plt.xlabel('$x$')
plt.ylabel('Probability density')
plt.xlim(0)
plt.ylim(0)
plt.show()

data = stats.beta.rvs(a, b, size=1000)
plt.hist(data, bins=20, normed=True)
plt.xlim(0, 1)
plt.title('Simulating Beta Random Variables:  \n$E(X)=%.2f,\quad D(X)=%.2f$' % (np.mean(data), np.std(data, ddof=1)))
plt.show()


# 2.3.7 威布尔分布
'''
威布尔分布（Weibull distribution）是可靠性分析和寿命检验的理论基础。
例如，可以使用此分布回答以下问题：
预计将在老化期间失效的项目所占的百分比是多少？例如，预计将在 8 小时老化期间失效的保险丝占多大百分比？
预计在有效寿命阶段有多少次保修索赔？例如，在该轮胎的 50,000 英里有效寿命期间预计有多少次保修索赔？
预计何时会出现快速磨损？例如，应将维护定期安排在何时以防止发动机进入磨损阶段？
'''

c = 1.79
x = np.arange(0.01, 1, 0.01)
y = stats.weibull_min.pdf(x, c)
plt.plot(x, y)
plt.title('Weibull: $f(x; c)=cx^{c-1}e^{-x^c}, \quad (c=%.2f)$' % c)
plt.xlabel('$x$')
plt.ylabel('Probability density')
plt.xlim(0)
plt.ylim(0)
plt.show()

data = stats.weibull_min.rvs(c, size=1000)
plt.hist(data, bins=20, normed=True)
plt.xlim(0, 3)
plt.title('Simulating Weibull Random Variables: \n$E(X)=%.2f,\quad SD(X)=%.2f$' % (np.mean(data), np.std(data, ddof=1)))
plt.show()


# 2.3.8 卡方分布
'''
卡方分布（chi-square distribution[2], χ²-distribution，或写作χ²分布）是概率论与统计学中常用的一种概率分布。
k个独立的标准正态分布变量的平方和服从自由度为k的卡方分布。
卡方分布是一种特殊的伽玛分布，是统计推断中应用最为广泛的概率分布之一，例如假设检验和置信区间的计算。
由卡方分布延伸出来皮尔森卡方检定常用于：
1. 样本某性质的比例分布与总体理论分布的拟合优度（例如某行政机关男女比是否符合该机关所在城镇的男女比）；
2. 同一总体的两个随机变量是否独立（例如人的身高与交通违规的关联性）；
3. 二或多个总体同一属性的同素性检定（意大利面店和寿司店的营业额有没有差距）。
'''

k = 55
x = np.arange(0.01, 1, 0.01)
y = stats.chi2.pdf(x, k)
plt.plot(x, y)
plt.title('$\chi^2$: $f(x; k)=\\frac{1}{2^{k/2}\Gamma(k/2)}x^{\\frac{k}{2}-1}e^{-x/2}, \quad (k=%i)$' % k)
plt.xlabel('$x$')
plt.ylabel('Probability density')
plt.xlim(0)
plt.ylim(0)
plt.show()

data = stats.chi2.rvs(k, size=1000)
plt.hist(data, bins=20, normed=True)
plt.xlim(0, 100)
plt.title('Simulating $\chi^2$ Random Variables: \n$E(X)=%.2f,\quad SD(X)=%.2f$' % (np.mean(data), np.std(data, ddof=1)))
plt.show()


# 2.3.9 F分布
'''
F-分布（F-distribution）是一种连续概率分布，被广泛应用于似然比率检验，特别是ANOVA中。
'''

dfn, dfd = 29, 18
x = np.arange(0.01, 1, 0.01)
y = stats.f.pdf(x, dfn, dfd)
plt.plot(x, y)
plt.title('$F$: $f(x; df_1, df_2)=\\frac{df_2^{df_2/2}df_1^{df_1/2}x^{df_1/2-1}}{(df_2+df_1x)^{(df_1+df_2)/2}B(df_1/2, df_2/2)}, \quad (df_1=%i,\ df_2=%i)$' % (dfn, dfd))
plt.xlabel('$x$')
plt.ylabel('Probability density')
plt.xlim(0)
plt.ylim(0)
plt.show()

data = stats.f.rvs(dfn, dfd, size=1000)
plt.hist(data, bins=20, normed=True)
plt.xlim(0, 5)
plt.title('Simulating $F$ Random Variables: \n$E(X)=%.2f,\quad SD(X)=%.2f$' % (np.mean(data), np.std(data, ddof=1)))
plt.show()
