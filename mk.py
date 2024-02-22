# Mann-Kendall突变点检测
# 数据序列y
# 结果序列UF，UB
#--------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
def Kendall_change_point_detection(index,inputdata,area_simple):
    inputdata = np.array(inputdata)
    n=inputdata.shape[0]
    # 正序列计算---------------------------------
    # 定义累计量序列Sk，初始值=0
    Sk             = [0]
    # 定义统计量UFk，初始值 =0
    UFk            = [0]
    # 定义Sk序列元素s，初始值 =0
    s              =  0
    Exp_value      = [0]
    Var_value      = [0]
    # i从1开始，因为根据统计量UFk公式，i=0时，Sk(0)、E(0)、Var(0)均为0
    # 此时UFk无意义，因此公式中，令UFk(0)=0
    for i in range(1,n):
        for j in range(i):
            if inputdata[i] > inputdata[j]:
                s = s+1
            else:
                s = s+0
        Sk.append(s)
        Exp_value.append((i+1)*(i+2)/4 )                     # Sk[i]的均值
        Var_value.append((i+1)*i*(2*(i+1)+5)/72 )            # Sk[i]的方差
        UFk.append((Sk[i]-Exp_value[i])/np.sqrt(Var_value[i]))
    # ------------------------------正序列计算
    # 逆序列计算---------------------------------
    # 定义逆序累计量序列Sk2，长度与inputdata一致，初始值=0
    Sk2             = [0]
    # 定义逆序统计量UBk，长度与inputdata一致，初始值=0
    UBk             = [0]
    UBk2            = [0]
    # s归0
    s2              =  0
    Exp_value2      = [0]
    Var_value2      = [0]
    # 按时间序列逆转样本y
    inputdataT = list(reversed(inputdata))
    # i从2开始，因为根据统计量UBk公式，i=1时，Sk2(1)、E(1)、Var(1)均为0
    # 此时UBk无意义，因此公式中，令UBk(1)=0
    for i in range(1,n):
        for j in range(i):
            if inputdataT[i] > inputdataT[j]:
                s2 = s2+1
            else:
                s2 = s2+0
        Sk2.append(s2)
        Exp_value2.append((i+1)*(i+2)/4 )                     # Sk[i]的均值
        Var_value2.append((i+1)*i*(2*(i+1)+5)/72 )            # Sk[i]的方差
        UBk.append((Sk2[i]-Exp_value2[i])/np.sqrt(Var_value2[i]))
        UBk2.append(-UBk[i])
    # 由于对逆序序列的累计量Sk2的构建中，依然用的是累加法，即后者大于前者时s加1，
    # 则s的大小表征了一种上升的趋势的大小，而序列逆序以后，应当表现出与原序列相反
    # 的趋势表现，因此，用累加法统计Sk2序列，统计量公式(S(i)-E(i))/sqrt(Var(i))
    #也不应改变，但统计量UBk应取相反数以表征正确的逆序序列的趋势
    #  UBk(i)=0-(Sk2(i)-E)/sqrt(Var)
    # ------------------------------逆序列计算
    # 此时上一步的到UBk表现的是逆序列在逆序时间上的趋势统计量
    # 与UFk做图寻找突变点时，2条曲线应具有同样的时间轴，因此
    # 再按时间序列逆转结果统计量UBk，得到时间正序的UBkT，
    UBkT = list(reversed(UBk2))
    diff = np.array(UFk) - np.array(UBkT)
    K    = list()
    # 找出交叉点
    for k in range(1,n):
        if diff[k-1]*diff[k]<0:
            K.append(k)
    # 做突变检测图时，使用UFk和UBkT

    fig=plt.figure(figsize=(8,4),dpi=300)
    ax = fig.add_subplot(111)
    bwith = 2 #边框宽度设置为2
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')



    fontdict={'family' : 'Times New Roman', 'size'   : 16}
    plt.plot(range(n, ) ,UFk  ,label='UF',color='red',marker='s') # UFk
    plt.plot(range(n, ) ,UBkT ,label='UB', color='blue', linestyle='--', marker='o') # UBk
    plt.ylabel('UFk-UBk')

    # 添加显著水平线和y=0
    # 添加辅助线
    x_lim = plt.xlim()
    plt.plot(x_lim,[-1.96,-1.96],':',color='black')
    plt.plot(x_lim,[  0  ,  0  ],'--',color='black')
    plt.plot(x_lim,[+1.96,+1.96],':',color='black')

    plt.xticks(range(n), index, fontdict=fontdict)
    plt.yticks(fontsize=11)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.ylabel('Mann-Kendall test value', fontdict=fontdict)
    plt.xlabel('Year', fontdict=fontdict)
    # plt.title(area_simple, fontdict=fontdict)
    plt.text(0.015, 0.97, area_simple, fontdict=fontdict, color='black',
             bbox=dict(facecolor='white', edgecolor='black'),
             transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left')

    x_major_locator=MultipleLocator(5)
    ax.xaxis.set_major_locator(x_major_locator)

    plt.show()
    return K

index=[1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]
input=[18.85047,18.621725,18.508102,18.497377,18.563808,18.671936,18.817886,19.058601,18.762302,18.703215,18.789518,18.714691,19.185438,18.915833,18.761465,18.87459,18.627039,18.714598,18.967249,18.914642,18.62189,18.766188,19.000858]

Kendall_change_point_detection(index,input,'AE')
