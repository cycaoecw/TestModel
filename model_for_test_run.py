#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 20:37:15 2018

@author: mo0301
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn import neural_network
import time
import tushare as ts
import talib as tb
import matplotlib.pyplot as plt

def get_price_change(close):
    close = np.array(close)
    close_2 = np.copy(close)
    close_2 = np.roll(close_2,1)
    
    result = close - close_2
    result[0] = 0
    return result

def get_p_change(close):
    close = np.array(close)
    close_2 = np.copy(close)
    close_2 = np.roll(close_2,1)
    
    result = (close - close_2)/close_2
    result[0] = 0
    result = result*100
    return result

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#     print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def get_grid_report(results, n_top=3):
    for i in range(1, n_top +1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank:  {0}".format(i))
            print('mean validation score: {0:.8f} (std: {1:.3f})'.format(
            results['mean_test_score'][candidate],
            results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def get_stock_date(symbol,s_date,e_date):
    ##############  在Tushare 获取股票数据   #######################
    code=symbol.split('.HS')[0]
    name=symbol.split('.HS')[1]
    print("Loading data...", code, '--', name)
    df=ts.get_k_data(code,start=s_date, end=e_date)
    df.set_index(['date'], inplace=True)
    df = df.drop(['code'],1)
    df = df.sort_index(ascending=True)  # 将数据按照日期排序下。
    ##############  在Tushare 获取股票数据     #######################
    return df

def data_cleaning(df_temp):
    #################################     计算各种指标           ##################################
    #################  close high low volume is "pandas.core.series.Series" #######################
    ################# output MACD_5 is"numpy.ndarray"  ############################################
    df=pd.DataFrame.copy(df_temp)
    close = df['close']
    volume= np.array(df['volume'],dtype='float64')
    high = df['high']
    low = df['low']

    ######################## input  is  "numpy.ndarray"  use .values to convert ###################
    ############## https://zhuanlan.zhihu.com/p/25407061  中文解析 Talib 指标使用 #################
    ma5 = tb.MA(close.values, timeperiod=5, matype=0)
    ma10 = tb.MA(close.values, timeperiod=10, matype=0)
    ma20 = tb.MA(close.values, timeperiod=20, matype=0)
    v_ma5 = tb.MA(volume, timeperiod=5, matype=0)
    v_ma10 = tb.MA(volume, timeperiod=10, matype=0)
    v_ma20 = tb.MA(volume, timeperiod=20, matype=0)
    price_change = get_price_change(close)
    p_change = get_p_change(close)
    MACD_5, MACD_Singal, hist = tb.MACD(close.values,fastperiod=12,slowperiod=26,signalperiod=9)
    EMA_12 = tb.EMA(close.values,timeperiod=12)
    EMA_5 = tb.EMA(close.values,timeperiod=5)
    EMA_20 = tb.EMA(close.values,timeperiod=20)
    RSI_6 = tb.RSI(close.values,timeperiod=6)
    RSI_12 = tb.RSI(close.values,timeperiod=12)
    SMA = tb.SMA(close.values,timeperiod=5)
    
    upper, middle, lower =tb.BBANDS(close.values,timeperiod=5,nbdevup=2,nbdevdn=2,matype=0)
    #matype:
    # 0    SMA – Simple Moving Average
    # 1    EMA – Exponential Average
    # 2    WMA – Weighted Moving Average
    # 3    DEMA – Double Exponential Moving Average
    # 4    TEMA – Triple Exponential Moving Average
    # 5    TRIMA – Triangular Moving Average
    # 6    KAMA – Kaufman Adaptive Moving Average
    # 7    MAMA – MESA Adaptive Moving Average
    # 8    T3 – Triple Exponential Moving Average
    
    KAMA = tb.KAMA(close.values,timeperiod=5)
    OBV = tb.OBV(close.values, volume)
    CCI = tb.CCI(high.values, low.values, close.values, timeperiod=5)
    DEMA = tb.DEMA(close.values,timeperiod=5)
    HT_TRENDLINE = tb.HT_TRENDLINE(close.values)
    MIDPOINT = tb.MIDPOINT(close.values,timeperiod=5)
    MIDPRICE = tb.MIDPRICE(high.values, low.values, timeperiod=5)
    CMO = tb.CMO(close.values, timeperiod=5)
    ADX =tb.ADX(high.values, low.values, close.values, timeperiod=5)
    ADXR = tb.ADXR(high.values, low.values, close.values, timeperiod=5)
    AROON_D, AROON_U = tb.AROON(high.values, low.values, timeperiod=5)
    ROC = tb.ROC(close.values, timeperiod=5)
    CMO = tb.CMO(close.values, timeperiod= 5)
    PPO = tb.PPO(close.values, fastperiod=3, slowperiod=5, matype=0)

    features=['ma5','ma10','ma20','v_ma5','v_ma10','v_ma20','price_change','p_change','EMA_5','EMA_12','EMA_20','MACD_5','MACD_Singal',
              'hist','RSI_6','RSI_12','SMA',
              'upper','middle','lower','KAMA','OBV','CCI','DEMA','HT_TRENDLINE','MIDPOINT','MIDPRICE',
              'CMO','ADX','ADXR','AROON_D','AROON_U','ROC','CMO','PPO']
    ######################计算各种指标#############################  
    
    ###### 将以上获得的features通过时间平移，将前五天的历史数据作为今天的features 建立一个235维的数据集。#####
    for f in features:
        df[f]=locals()[f]
    indicators = list(df.columns.values)
    for i in range(1,6):
        for indicator in indicators:
            name= "{0}".format(i) +'_'+indicator
            df[name] = df[indicator].shift(i)   
    df.dropna(inplace=True)   #只要有NaN的行，全部删出以免出错
    print('Cleaning Completed')
    return df

def get_train_test(df, size =50):
    ##########  设置labels，对比当日和未来3天收盘价中的最高者，并做 (< -5%), (>= -5%, 0%), (>=0%, 3%), (>=3%)的分类
    ########## (< -5%) :         label 0
    ##########  (>= -5%, 0%): label 1
    ##########  (>=0%, 3%):    label 2
    ##########  (>=3%):          label 3
    max_value = np.array([])
    for i in range(0, len(df)-3):
        a = np.array([df['close'].iloc[i+1], df['close'].iloc[i+2], df['close'].iloc[i+3]])
        max_value = np.append(max_value, [max(a)])  
    for i in range(0,3):
        max_value = np.append(max_value,0)
    df['max_value']=max_value
    label= np.array(df['max_value']/df['close']-1)
    label = np.where((label>=0.03),3,label)
    label = np.where((label<0.03) & (label>=0.0),2,label)
    label = np.where((label<0.0) & (label>=-0.05),1,label)
    label = np.where((label<-0.05), 0,label)
        
    df['label']=label
    X = np.array(df.drop(['label'], 1))   #去掉label的那一列
    X = preprocessing.scale(X)          #正则化数据    
    X = X[:-3]               #抽出除了最后1行
    date = df.index.values
    df = df.drop(df.index[range(len(df)-3 , len(df))], axis=0)
    y = np.array(df[['label']])    
    ############建立预测模型###########
    print('Model data end at:' ,date[-size:][0])
    X_train, X_test, y_train ,y_test = train_test_split(X, y, test_size=size, shuffle=False)
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    
    return X_train, X_test, y_train, y_test

def get_model(X_train, y_train):
    start_time = time.time()
    print('Building model please wait...')
    model = neural_network.MLPClassifier(solver='lbfgs', 
                                         max_iter= 500, batch_size=50, hidden_layer_sizes=1000,warm_start=True)
    search_params = {'learning_rate' : ['constant', 'invscaling'],
                            'activation':['identity', 'relu']}  
    grid_search = GridSearchCV(model , search_params, n_jobs=-1)
    
    grid_search.fit(X_train, y_train)  
#     joblib.dump(grid_search, 'grid_model.pkl')
    get_grid_report(grid_search.cv_results_)
    print("Model has been exported")
    print('Model is built, total time useage %.1f seconds.' %(time.time()-start_time))
    return grid_search



######################       the following coding is execution   ######################
temp = pd.read_csv('test to run.csv', encoding='gbk')
pool = np.array(temp['code'].values)

date_start = "2006-12-01"
date_end = "2018-07-13"

for A in pool:
    s_time=time.time()

    stock_raw =get_stock_date(A,date_start,date_end)
    data_clean = data_cleaning(stock_raw)
    X_train, X_test, y_train, y_test =  get_train_test(data_clean)

    model = get_model(X_train, y_train)
    y_pred = model.predict(X_test)

    # Plot non-normalized confusion matrix
    c_matrix = confusion_matrix(y_test, y_pred, labels=[0,1,2,3])
    class_names=np.array(["less -5%","-5% to 0%","0%-3%","larger 3%"])
    plt.figure()
    plot_confusion_matrix(c_matrix, classes=class_names,title='Confusion matrix, without normalization')
    print('Accuracy: ', round(sum(np.diag(c_matrix))/sum(sum(c_matrix)),4)*100,'%')
    print('Total run time %.1f seconds.' %(time.time()-s_time))
    print('------------------------------------------------------------------------------')