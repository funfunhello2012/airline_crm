# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 19:31:25 2019

@author: wufan
"""
import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#导入数据
df = pd.read_csv('C:/Git/python_data_analysis_and_mining_action/chapter7/data/air_data.csv', \
                 encoding='utf-8')
df_a=df.copy(deep=True)
#探索数据
explore = df_a.describe(percentiles=[], include='all').T
# describe()函数自动计算非空值数，需要手动计算空值数
explore['null'] = len(df_a) - explore['count']

explore = explore[['null', 'max', 'min']]
explore.columns = [u'空值数', u'最大值', u'最小值']
#数据清洗    
a=[]
for i in range(len(df_a)):
    load_time=datetime.strptime(df['LOAD_TIME'][i], '%Y/%m/%d')
    temp=datetime.strptime(df_a['FFP_DATE'][i], '%Y/%m/%d')
    a.append(round((load_time-temp).days/30))
df_a['MENBER_TIME']=a
df_a = df_a[df_a['SUM_YR_1'].notnull() & df_a['SUM_YR_2'].notnull()]
df_a = df_a[df_a['SUM_YR_1'] != 0]
df_a = df_a[df_a['SUM_YR_2'] != 0]
df_a = df_a[df_a['SEG_KM_SUM'] != 0]
df_a = df_a[df_a['avg_discount'] != 0]
df_a.dropna(subset=['WORK_PROVINCE'], inplace=True)
df_a.dropna(inplace=True)
#再一次数据探索，确认没有不合理的值和缺失值
explore = df_a.describe(percentiles=[], include='all').T
# describe()函数自动计算非空值数，需要手动计算空值数
explore['null'] = len(df_a) - explore['count']
explore = explore[['null', 'max', 'min']]
explore.columns = [u'空值数', u'最大值', u'最小值']
#L＝LOAD_TIME－FFP_DATE=MENBER_TIME 会员入会时间距观测窗口结束的月数＝观测窗口的结束时间－入会时间[单位：月]
#R＝LAST_TO_END 客户最近一次乘坐公司飞机距观测窗口结束的月数＝最后一次乘机时间至观察窗口末端
#F＝FLIGHT_COUNT 客户在观测窗口内乘坐公司飞机的次数＝观测窗口的飞行次数[单位：次]
#M＝SEG_KM_SUM 客户在观测时间内在公司累计的飞行里程＝观测窗口的总飞行公里数[单位：公里]
#C＝AVG_DISCOUNT 客户在观测时间内乘坐舱位所对应的折扣系数的平均值＝平均折扣率[单位：无]
features = ['MENBER_TIME', 'LAST_TO_END', 'FLIGHT_COUNT', 'SEG_KM_SUM', 'avg_discount']
df_b = df_a[features]
df_b.rename(columns={'avg_discount':'AVG_DISCOUNT'},inplace=True)
ss = StandardScaler()
df_b = ss.fit_transform(df_b)
    
    
kmodel = KMeans(n_clusters=5)
kmodel.fit(df_b)
print(kmodel.cluster_centers_)  # 查看聚类中心
print(kmodel.labels_)  # 查看各样本对应的类别

fig = plt.figure(figsize=[7, 5])
clu = kmodel.cluster_centers_
x = [1,2,3,4,5]
colors = ['red','green','yellow','blue','black']
for i in range(5):
    plt.plot(x, clu[i],label='cluster'+str(i)+' '+str(cluster_value[i]), color=colors[i], marker='o')
plt.legend()
plt.xlabel('L R F M C')
plt.ylabel('values')
plt.show()

cluster_list=list(kmodel.labels_)
cluster_value=pd.value_counts(cluster_list)