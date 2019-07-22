# -*- encoding: utf-8 -*-
import numpy as np
import pandas as pd


data = [1, 3, 5, 6, 7, 9]
df = pd.DataFrame(data, columns=['data'])

# 1 数据的描述性统计
# 1.1 数据的集中趋势
# 1.1.1 众数
'''
众数（Mode）是统计学名词，在统计分布上具有明显集中趋势点的数值，
一组数据中出现次数最多的数值，代表数据的一般水平（众数可以不存在或多于一个）。
The mode of a set of values is the value that appears most often.
It can be multiple values.
'''


# python自定义函数
def get_mode(data, all=True):
    if data:
        value_count = {}
        for i in data:
            if value_count.get(i):
                value_count[i] += 1
            else:
                value_count[i] = 1
        if all is True:
            mode = [k for k, v in value_count.items() if v == max(value_count.values())]
        elif all is False:
            mode = max(value_count, key=value_count.get)
        else:
            raise TypeError(all)
    else:
        mode = None
    return mode


# python使用pandas包
df.mode()

# hql实现
'''
select data mode_data
from (
  select data, num, row_number() over(order by num desc) rn
  from (
    select data, count(*) num
    from temp.descriptive_statistics_of_data
    group by data
  ) t1
) t2
where rn = 1
'''


# 1.1.2 中位数
'''
中位数（又称中值，英语：Median），代表一个样本、种群或概率分布中的一个数值，其可将数值集合划分为相等的上下两部分。
对于有限的数集，可以通过把所有观察值高低排序后找出正中间的一个作为中位数。
如果观察值有偶数个，则中位数不唯一，通常取最中间的两个数值的平均数作为中位数。
The median is the value separating the higher half from the lower half of a data sample 
(a population or a probability distribution).
For a data set, it may be thought of as the "middle" value.
'''


# python自定义函数
def get_median(data):
    if data:
        data.sort()
        half = len(data) // 2
        median = (data[half] + data[~half]) / 2
    else:
        median = None
    return median


# python使用numpy包
np.median(data)
# python使用pandas包
df.median()

# hql实现
'''
select percentile(data, 0.5) median_data
from temp.descriptive_statistics_of_data
'''


# 1.1.3 平均数
'''
平均数（英语：Mean，或称平均值）是统计中的一个重要概念。为集中趋势的最常用测度值，目的是确定一组数据的均衡点。
不是所有类型的资料都能使用平均数，在没有充分考虑个体和群体性质的状况下，平均数可以得出毫无意义或无法反映现实的结果，
像是例如说“男人和女人平均有一颗睾丸”[note 1]就是正确但无意义的平均数。
For a data set, the arithmetic mean, also called the mathematical expectation or average, 
is the central value of a discrete set of numbers: 
specifically, the sum of the values divided by the number of values.
'''


# python自定义函数
def get_mean(data, num=2):
    if data:
        mean = round(sum(data) / len(data), num)
    else:
        mean = None
    return mean


# python使用numpy包
np.mean(data)
# python使用pandas包
df.mean()

# hql实现
'''
select avg(data) avg_data
from temp.descriptive_statistics_of_data
'''

# 1.1.4 分位数
'''
分位数（Quantile），亦称分位点，是指将一个随机变量的概率分布范围分为几个等份的数值点，常用的有中位数（即二分位数）、四分位数、百分位数等。
1.二分位数
对于有限的数集，可以通过把所有观察值高低排序后找出正中间的一个作为中位数。
如果观察值有偶数个，则中位数不唯一，通常取最中间的两个数值的平均数作为中位数，即二分位数。
一个数集中最多有一半的数值小于中位数，也最多有一半的数值大于中位数。
如果大于和小于中位数的数值个数均少于一半，那么数集中必有若干值等同于中位数。
计算有限个数的数据的二分位数的方法是：把所有的同类数据按照大小的顺序排列。
如果数据的个数是奇数，则中间那个数据就是这群数据的中位数；如果数据的个数是偶数，则中间那2个数据的算术平均值就是这群数据的中位数。
2.四分位数
四分位数（Quartile）是统计学中分位数的一种，即把所有数值由小到大排列并分成四等份，处于三个分割点位置的数值就是四分位数。
1）第一四分位数(Q1)，又称“较小四分位数”，等于该样本中所有数值由小到大排列后第25%的数字；
2）第二四分位数(Q2)，又称“中位数”，等于该样本中所有数值由小到大排列后第50%的数字；
3）第三四分位数(Q3)，又称“较大四分位数”，等于该样本中所有数值由小到大排列后第75%的数字。
第三四分位数与第一四分位数的差距又称四分位距。
3.百分位数
百分位数，统计学术语，如果将一组数据从小到大排序，并计算相应的累计百分位，则某一百分位所对应数据的值就称为这一百分位的百分位数。
运用在教育统计学中，例如表现测验成绩时，称PR值。

'''


# python自定义函数
def get_quantile(data, p):
    pos = (len(data) - 1) * p
    pos_integer = int(np.floor(pos))
    pos_decimal = pos - pos_integer
    quantile = data[pos_integer] * (1 - pos_decimal) + data[pos_integer + 1] * pos_decimal
    return quantile


# python使用numpy包
# 25%分位数，第一四分位数
np.percentile(data, 25)
# python使用pandas包
# 二分位数，中位数
df.quantile(1/2)

# hql实现
# percentile求精确的百分位数，要求数据必须是整型
'''
select percentile(cast(data as bigint), 0.25) quantile_data
from temp.descriptive_statistics_of_data
'''
# percentile_approx求近似的百分位数，数据可以是浮点型等
'''
select percentile_approx(data, 0.25) quantile_data
from temp.descriptive_statistics_of_data
'''

# 1.1.5 极差
'''
极差又称范围误差或全距(Range)，以R表示，是用来表示统计资料中的变异量数(measures of variation)，
最大值与最小值之间的差距，即最大值减最小值后所得之数据。
'''


# python自定义函数
def get_range(data):
    range = max(data) - min(data)
    return range


# python使用numpy包
np.max(data) - np.min(data)
# python使用pandas包
df.max() - df.min()

# hql实现
'''
select max(data) - min(data) data_range
from temp.descriptive_statistics_of_data
'''

# 1.1.6 算术平均数
'''
算术平均数（ arithmetic mean），又称均值，是统计学中最基本、最常用的一种平均指标，分为简单算术平均数、加权算术平均数。
它主要适用于数值型数据，不适用于品质数据。根据表现形式的不同，算术平均数有不同的计算形式和计算公式。
算术平均数是加权平均数的一种特殊形式（特殊在各项的权重相等）。
在实际问题中，当各项权重不相等时，计算平均数时就要采用加权平均数；当各项权相等时，计算平均数就要采用算术平均数。
'''
# 代码实现见1.1.3 平均数

# 1.1.7 加权平均数
'''
加权平均值即将各数值乘以相应的权数，然后加总求和得到总体值，再除以总的单位数。
加权平均值的大小不仅取决于总体中各单位的数值（变量值）的大小，而且取决于各数值出现的次数（频数），
由于各数值出现的次数对其在平均数中的影响起着权衡轻重的作用，因此叫做权数。
'''

# python使用numpy包
# 产生一个介于[0, 10)的随机权重
weight = np.random.randint(10, size=len(data))
np.average(data, weights=weight)
# python使用pandas包
df['weight'] = pd.DataFrame(np.random.randint(10, size=len(data)))
sum(df.data * df.weight) / sum(df.weight)

# hql实现
'''
select sum(data*weight)/sum(weight) weighted_avg_data
from temp.descriptive_statistics_of_data
'''

# 1.1.7 几何平均数
'''
几何平均数是对各变量值的连乘积开项数次方根。
如果总水平、总成果等于所有阶段、所有环节水平、成果的连乘积总和时，求各阶段、各环节的一般水平、一般成果，
要使用几何平均法计算几何平均数，而不能使用算术平均法计算算术平均数。
分为简单几何平均数和加权几何平均数两种形式。
'''
# python使用numpy包
np.power(np.array(data).prod(), 1 / len(data))

# hql实现
'''
select pow(exp(sum(ln(data))), 1/count(data))
from temp.descriptive_statistics_of_data
'''

# 1.2 数据的离中趋势
# 1.2.1 数值型数据：方差
'''
方差是在概率论和统计方差衡量随机变量或一组数据时离散程度的度量。
概率论中方差用来度量随机变量和其数学期望（即均值）之间的偏离程度。
'''


# python自定义函数
# 总体方差
def get_var(data):
    mean = get_mean(data)
    var = sum([np.power(x - mean, 2) for x in data])/len(data)
    return var


# python使用numpy包
# 总体方差
np.var(data)
# 样本方差
np.var(data, ddof=1)

# python使用pandas包
# 总体方差
df.var(ddof=0)
# 样本方差
df.var()

# hql实现
# 总体方差
'''
select variance(data) data_var
from temp.descriptive_statistics_of_data
'''

# 1.2.2 数值型数据：标准差
'''
标准差（Standard Deviation） ，中文环境中又常称均方差，是离均差平方的算术平均数的平方根，用σ表示。
标准差是方差的算术平方根。标准差能反映一个数据集的离散程度。
平均数相同的两组数据，标准差未必相同。
'''


# python自定义函数
def get_stddev(data):
    mean = get_mean(data)
    stddev = np.sqrt(sum([np.power(x - mean, 2) for x in data])/len(data))
    return stddev


# python使用numpy包
# 总体标准差
np.std(data)
# 样本标准差
np.std(data, ddof=1)

# python使用pandas包
# 总体标准差
df.std(ddof=0)
# 样本标准差
df.std()

# hql实现
# data_stddev为总体标准差，data_stddev_samp为样本标准差
'''
select stddev(data) data_stddev, stddev_samp(data) data_stddev_samp
from temp.descriptive_statistics_of_data
'''

# 1.2.3 数值型数据：极差
# 见 1.1.5 极差

# 1.2.4 数值型数据：平均差
'''
平均差（Mean Deviation）是表示各个变量值之间差异程度的数值之一。指各个变量值同平均数的离差绝对值的算术平均数。
平均差异大，表明各标志值与算术平均数的差异程度越大，该算术平均数的代表性就越小；
平均差越小，表明各标志值与算术平均数的差异程度越小，该算术平均数的代表性就越大。
平均差是反应各标志值与算术平均数之间的平均差异。
'''


# python自定义函数
def get_avgdev(num_list):
    mean = get_mean(data)
    avgdev = sum([abs(x-mean) for x in data])/len(data)
    return avgdev


# hql实现
'''
select sum(abs(data - avg_data))/count(data) data_avgdev
from (
  select data, avg(data) over(partition by data) avg_data
  from temp.descriptive_statistics_of_data
  ) t
'''

# 1.2.5 顺序数据：四分位差
'''
四分位差（quartile deviation），它是上四分位数（Q3，即位于75%）与下四分位数（Q1，即位于25%）的差。计算公式为：Q = Q3-Q1
四分位差反映了中间50%数据的离散程度，其数值越小，说明中间的数据越集中；其数值越大，说明中间的数据越分散。
四分位差不受极值的影响。
此外，由于中位数处于数据的中间位置，因此，四分位差的大小在一定程度上也说明了中位数对一组数据的代表程度。
'''


# python自定义函数
def get_quartile_dev(data):
    quartile_dev = get_quantile(data, 0.75) - get_quantile(data, 0.25)
    return quartile_dev


# python使用numpy包
# 25%分位数，第一四分位数
np.percentile(data, 75) - np.percentile(data, 25)
# python使用pandas包
df.quantile(0.75) - df.quantile(0.25)

# hql实现
'''
select percentile(cast(data as bigint), 0.75) - percentile(cast(data as bigint), 0.25) quantile_data
from temp.descriptive_statistics_of_data
'''

# 1.2.6 分类数据：异众比率
'''
异众比率（variation ratio）是统计学名词，是统计学当中研究现象离中趋势的指标之一。
异众比率指的是总体中非众数次数与总体全部次数之比。换句话说，异众比率指非众数组的频数占总频数的比例。
'''


# python自定义函数
def get_var_ratio(data):
    count = 0
    mode_data = get_mode(data)
    for x in data:
        if x not in mode_data:
            count += 1
    return count/len(data)


# 1.3 相对离散程度
# 1.3.1 离散系数
'''
离散系数又称变异系数，是统计学当中的常用统计指标。
离散系数是测度数据离散程度的相对统计量，主要是用于比较不同样本数据的离散程度。
离散系数大，说明数据的离散程度也大；离散系数小，说明数据的离散程度也小。
'''


# python自定义函数
def get_coef_of_var(data):
    coef_of_var = get_stddev(data)/get_mean(data)
    return coef_of_var


# hql实现
'''
select stddev(data)/avg(data) coef_of_var
from temp.descriptive_statistics_of_data
'''

# 1.4 分布的形状
# 1.4.1 偏态系数
'''
偏度（skewness）也称为偏态、偏态系数(deviation coefficient)，是统计数据分布偏斜方向和程度的度量，是统计数据分布非对称程度的数字特征。
偏态系数绝对值越大，偏斜越严重。
'''


# python自定义函数
def get_skew(data):
    skewness = np.mean((data - np.mean(data)) ** 3)
    return skewness


# python使用pandas包
df.skew()

# hql实现
'''
select round(sum(power((data-avg_data),3))/power(stddev(data),3)/(count(*)-1),2) as skewness 
from (
  select data, avg(data) over(partition by data) avg_data
  from temp.descriptive_statistics_of_data
  ) t
'''

# 1.4.2 峰态系数
'''
峰度（peakedness;kurtosis）又称峰态系数(Coefficient of kurtosis)，是用来反映频数分布曲线顶端尖峭或扁平程度的指标。
表征概率密度分布曲线在平均值处峰值高低的特征数。
直观看来，峰度反映了峰部的尖度。
样本的峰度是和正态分布相比较而言统计量，如果峰度大于三，峰的形状比较尖，比正态分布峰要陡峭。反之亦然。
'''


# python自定义函数
def get_kurt(data):
    kurtosis = np.mean((data - np.mean(data)) ** 4) / pow(np.var(data), 2)
    return kurtosis


# python使用pandas包
df.kurt()

# hql实现
'''
select round(sum(power((data - avg_data),4))/power(stddev(data),4)/(count(*)-1)-3,2) as kurtosis 
from (
  select data, avg(data) over(partition by data) avg_data
  from temp.descriptive_statistics_of_data
  ) t
'''
