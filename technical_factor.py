import datetime
import os.path
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import tushare as ts
import mplfinance as mpf
pd.set_option('mode.chained_assignment', None)

class Technical_Factor:
    # def __init__(self):
        # self.data = data
        # self.data_weekly = data.resample('w').last()
        # self.data_monthly = data.resample('M').last()

    def data_rename(self, dat):
        # tp = dat.copy()
        dat.rename(columns={'ts_code': 'ts_code', 'close': 'PX_LAST', 'open': 'PX_OPEN', 'high': 'PX_HIGH', 'low': 'PX_LOW',
                 'vol': 'PX_VOLUME'}, inplace=True)
        return dat

    def MA(self, dat, n):
        return dat.rolling(n).mean()
    def EMA(self, dat, n):
        return dat.ewm(span=n).mean()
    def STD(self, dat, n):
        return dat.rolling(n).std()
    def divergence(self, dat, n, date, factor):
        '''
        判断指标背离情况
        :return: 1：顶背离；-1：底背离； 0：无背离
        '''
        if dat.loc[date, 'PX_LAST'] > dat.shift(n).loc[date, 'PX_LAST'] \
                and dat.loc[date, factor] < dat.shift(n).loc[date, factor]:
            return 1
        elif dat.loc[date, 'PX_LAST'] < dat.shift(n).loc[date, 'PX_LAST'] \
                and dat.loc[date, factor] > dat.shift(n).loc[date, factor]:
            return -1
        else:
            return 0
    def interval(self, dat, a, b):
        tp = dat[dat > a]
        return tp[tp <= b].index

    def ROC(self, dat, n=10):
        roc = (dat['PX_LAST'].shift(1) - dat['PX_LAST'].shift(n)) / dat['PX_LAST'].shift(n) * 100
        # dat['ROC'] = roc
        return roc
    def ROC__(self, dat, n=10):
        dat['ROC'] = self.ROC(dat, n)
        dat['ROC__'] = 0
        dat.loc[dat['ROC'] > 0, 'ROC__'] = 1
        dat.loc[dat['ROC'] < 0, 'ROC__'] = -1
        return dat
    def ROC_(self, dat, n=10):
        '''
        背离用法，卖点精确，买点很少
        '''
        dat['ROC'] = self.ROC(dat, n)
        dat['ROC_'] = 0
        # hold = 0
        # for date in dat.index:
        #     if hold == 0 and self.divergence(dat, n, date, 'ROC') == -1:
        #         dat.loc[date, 'ROC__'] = 1
        #         hold = 1
        #     elif hold == 1 and self.divergence(dat, n, date, 'ROC') == 1:
        #         dat.loc[date, 'ROC__'] = -1
        #         hold == 0
        #     else:
        #         dat.loc[date, 'ROC__'] = 0
        # dat.loc[dat['ROC'] > 0, 'ROC__'] = 1
        for date in dat.index:
            if self.divergence(dat, n, date, 'ROC') == -1:
                dat.loc[date, 'ROC_'] = 1
            elif self.divergence(dat, n, date, 'ROC') == 1:
                dat.loc[date, 'ROC_'] = -1
        return dat


    def MA_(self, dat, n=20):
        dat['MA'] = self.MA(dat['PX_LAST'], n)
        dat['MA_'] = 0
        dat.loc[dat['MA'] > dat['PX_LAST'], 'MA_'] = -1
        dat.loc[dat['MA'] < dat['PX_LAST'], 'MA_'] = 1
        return dat

    def SMA(self, dat, n1=10, n2=20):
        sma = self.MA(dat['PX_LAST'], n1)
        lma = self.MA(dat['PX_LAST'], n2)
        Sma = sma - lma
        return Sma
    def SMA_(self, dat, n1=10, n2=20):
        dat['SMA'] = self.SMA(dat, n1, n2)
        dat['SMA_'] = 0
        dat.loc[dat['SMA'] > 0, 'SMA_'] = 1
        dat.loc[dat['SMA'] < 0, 'SMA_'] = -1
        return dat

    def DMA(self, dat, n1=10, n2=50):
        dma = self.MA(dat['PX_LAST'], n1) - self.MA(dat['PX_LAST'], n2)
        ama = self.MA(dma, n1)
        return dma, ama
    def DMA__(self, dat, n1=10, n2=50):
        dma, ama = self.DMA(dat, n1, n2)
        dat['DMA'] = dma
        dat['AMA'] = ama
        dat['DMA_'] = 0
        for date in dat.index:
            if dat.loc[date, 'DMA']>0 and dat.loc[date, 'AMA']>0:
                dat.loc[date, 'DMA_'] = 1
            elif dat.loc[date, 'DMA']<0 and dat.loc[date, 'AMA']<0:
                dat.loc[date, 'DMA_'] = -1
        return dat
    def DMA_(self, dat, n1=10, n2=50):
        dma, ama = self.DMA(dat, n1, n2)
        dat['DMA'] = dma
        dat['AMA'] = ama
        dat['DMA_'] = 0
        for date in dat.index:
            if self.divergence(dat, n1, date, 'DMA') == 1 \
                    and self.divergence(dat, n1, date, 'AMA') == 1:
                if dat.loc[date, 'DMA'] > dat.loc[date, 'AMA']:
                    dat.loc[date, 'DMA_'] = 1
            elif self.divergence(dat, n1, date, 'DMA') == -1 \
                    and self.divergence(dat, n1, date, 'AMA') == -1:
                if dat.loc[date, 'DMA'] < dat.loc[date, 'AMA']:
                    dat.loc[date, 'DMA_'] = -1
        return dat

    def MACD(self, dat, short=12, long=26, mmid=9):
        dif = self.EMA(dat['PX_LAST'], short) - self.EMA(dat['PX_LAST'], long)
        dea = self.MA(dif, mmid)
        macd = 2*(dif-dea)
        return dif, dea, macd
    def MACD__(self, dat, short=12, long=26, mmid=9):
        dat['DIF'], dat['DEA'], dat['MACD'] = self.MACD(dat, short, long, mmid)
        dat['MACD_'] = 0
        dat.loc[dat['MACD'] > 0, 'MACD_'] = 1
        dat.loc[dat['MACD'] < 0, 'MACD_'] = -1
        # for date in dat.index:
        #     if dat.loc[date, 'MACD'] > 0 and dat.loc[date, 'DIF'] > dat.loc[date, 'DEA']:
        #         dat.loc[dat['MACD'] > 0, 'MACD_'] = 1
        #     elif dat.loc[date, 'MACD'] < 0 and dat.loc[date, 'DIF'] < dat.loc[date, 'DEA']:
        #         dat.loc[dat['MACD'] > 0, 'MACD_'] = -1
        return dat
    def MACD_(self, dat, short=12, long=26, mmid=9):
        '''
        背离用法，严格买，宽松卖
        '''
        dat['DIF'], dat['DEA'], dat['MACD'] = self.MACD(dat, short, long, mmid)
        dat['MACD_'] = 0
        for date in dat.index:
            if self.divergence(dat, mmid, date, 'MACD') == -1 \
                    and self.divergence(dat, mmid, date, 'DIF') == -1:
                dat.loc[date, 'MACD_'] = 1
            elif self.divergence(dat, mmid, date, 'MACD') == 1 \
                    or self.divergence(dat, mmid, date, 'DIF') == 1:
                dat.loc[date, 'MACD_'] = -1
        return dat


    def TRIX(self, dat, n=12, m=9):
        tr = self.EMA(self.EMA(self.EMA(dat['PX_LAST'], n), n), n)
        trix = (tr-tr.shift(1)) / tr.shift(1) * 100
        matrix = self.MA(trix, m)
        return trix, matrix
    def TRIX_(self, dat, n=12, m=9):
        dat['TRIX'], dat['MATRIX'] = self.TRIX(dat, n, m)
        dat['TRIX_'] = 0
        dat.loc[dat['TRIX'] > dat['MATRIX'], 'TRIX_'] = 1
        dat.loc[dat['TRIX'] < dat['MATRIX'], 'TRIX_'] = -1
        return dat

    def BBI(self, dat, n1=3, n2=6, n3=12, n4=24):
        close = dat['PX_LAST']
        bbi = (self.MA(close, n1)+self.MA(close, n2)+self.MA(close, n3)+self.MA(close, n4)) / 4
        return bbi
    def BBI_(self, dat, n1=3, n2=6, n3=12, n4=24):
        dat['BBI'] = self.BBI(dat, n1, n2, n3, n4)
        dat['BBI_'] = 0
        for date in dat.index:
            if self.divergence(dat, 3, date, 'BBI') == -1 \
                    and dat.loc[date, 'PX_LAST'] > dat.loc[date, 'BBI']:
                dat.loc[date, 'BBI_'] = 1
            elif self.divergence(dat, 3, date, 'BBI') == 1 \
                    or dat.loc[date, 'PX_LAST'] < dat.loc[date, 'BBI']:
                dat.loc[date, 'BBI_'] = -1
        return dat
    def BBI__(self, dat, n1=3, n2=6, n3=12, n4=24):
        dat['BBI'] = self.BBI(dat, n1, n2, n3, n4)
        dat['BBI_'] = 0
        dat.loc[dat['PX_LAST'] > dat['BBI'], 'BBI_'] = 1
        dat.loc[dat['PX_LAST'] < dat['BBI'], 'BBI_'] = -1
        dat.loc[dat['BBI'] == None] = 0
        return dat

    def BOLL(self, dat, n=20, p=2):
        mb = self.MA(dat['PX_LAST'], n)
        draft = p*dat['PX_LAST'].rolling(n).std()
        up = mb+draft
        down = mb-draft
        return mb, up, down
    def BOLL_(self, dat, n=20, p=2):
        dat['BOLL_MB'], dat['BOLL_UP'], dat['BOLL_DOWN'] = self.BOLL(dat, n, p)
        dat['BOLL_'] = 0
        dat.loc[dat['PX_LAST'] > dat['BOLL_UP'], 'BOLL_'] = 1
        dat.loc[dat['PX_LAST'] < dat['BOLL_DOWN'], 'BOLL_'] = -1
        return dat

    def Aberration_(self, dat, n=20, p=2):
        dat['BOLL_MB'], dat['BOLL_UP'], dat['BOLL_DOWN'] = self.BOLL(dat, n, p)
        dat['Aberration_'] = 0
        dat.loc[dat['PX_LAST'] > dat['BOLL_UP'], 'Aberration_'] = 1
        dat.loc[dat['PX_LAST'] <= dat['BOLL_MB'], 'Aberration_'] = -1
        return dat

    def BIAS(self, dat, n=10):
        bias = (dat['PX_LAST'] / self.MA(dat['PX_LAST'], n) - 1) * 100
        return bias
    def BIAS_(self, dat, n=10):
        dat['BIAS'] = self.BIAS(dat, n)
        dat['BIAS_'] = 0
        dat.loc[dat['BIAS'] > 20, 'BIAS_'] = 1
        dat.loc[dat['BIAS'] < 20, 'BIAS_'] = -1
        return dat

    def CCI(self, dat, n=14):
        tp = (dat['PX_HIGH']+dat['PX_LOW']+dat['PX_LAST']) / 3
        ma = sum(tp.shift(i) for i in range(n+1)) / n
        md = sum(abs(ma-tp).shift(i) for i in range(n+1)) / n
        cci = (tp-ma) / (0.015*md)
        return cci
    def CCI_(self, dat, n=14):
        dat['CCI'] = self.CCI(dat, n)
        dat['CCI_'] = 0
        for date in dat.index:
            if dat.loc[date, 'CCI'] > -100 and dat.shift(1).loc[date, 'CCI'] < -100:
                dat.loc[date, 'CCI_'] = 1
            elif dat.loc[date, 'CCI'] < 100 and dat.shift(1).loc[date, 'CCI'] > 100:
                dat.loc[date, 'CCI_'] = -1
        return dat

    def KDJ(self, dat, n=9, m1=3, m2=3):
        ln = pd.DataFrame(index=dat.index)
        hn = pd.DataFrame(index=dat.index)
        for date in dat.index:
            ln.loc[date, 'ln'] = min(dat.loc[date-datetime.timedelta(days=n):date, 'PX_LAST'])
            hn.loc[date, 'hn'] = max(dat.loc[date-datetime.timedelta(days=n):date, 'PX_LAST'])
        rsv = 100 * (dat['PX_LAST'] - ln['ln']) / (hn['hn'] - ln['ln'])
        k = [0] * len(dat['PX_LAST'])
        d = [0] * len(dat['PX_LAST'])
        k[0] = 50
        d[0] = 50
        k = pd.Series(k, index=dat.index)
        d = pd.Series(d, index=dat.index)
        k = (m1 - 1) / m1 * k.shift(1) + 1 / m1 * rsv
        d = (m2 - 1) / m2 * d.shift(1) + 1 / m2 * k
        j = 3 * k - 2 * d
        return k, d, j


    def KDJ__(self, dat, n=9, m1=3, m2=3):
        dat['KDJ_K'], dat['KDJ_D'], dat['KDJ_J'] = self.KDJ(dat, n, m1, m2)
        dat['KDJ_'] = 0
        dat.loc[dat['KDJ_K'] > dat['KDJ_D'], 'KDJ_'] = 1
        dat.loc[dat['KDJ_K'] < dat['KDJ_D'], 'KDJ_'] = -1
        return dat
    def KDJ_(self, dat, n=9, m1=3, m2=3):
        dat['KDJ_K'], dat['KDJ_D'], dat['KDJ_J'] = self.KDJ(dat, n, m1, m2)
        dat['KDJ_'] = 0
        for date in dat.index:
            if dat.loc[date, 'KDJ_K'] < 20 or (dat.loc[date, 'KDJ_K'] > 50 and dat.loc[date, 'KDJ_K'] < 80):
                if dat.loc[date, 'KDJ_J'] > dat.loc[date, 'KDJ_K']\
                    and dat.loc[date, 'KDJ_K'] > dat.loc[date, 'KDJ_D']:
                    dat.loc[date, 'KDJ_'] = 1
            elif dat.loc[date, 'KDJ_K'] > 80:
                if dat.loc[date, 'KDJ_J'] < dat.loc[date, 'KDJ_K'] \
                        or dat.loc[date, 'KDJ_K'] < dat.loc[date, 'KDJ_D']:
                    dat.loc[date, 'KDJ_'] = -1
        return dat

    def RHL_(self, dat):
        def R(r):
            R = self.EMA(dat['PX_LAST'], r)
            # R = self.EMA((dat['PX_LAST'] + dat['PX_LOW'] + dat['PX_HIGH'])/3, r)
            return R
        def H(r, h):
            H = self.EMA(R(r), h)
            return H
        L = 3*R(1) - 2*H(1, 2)
        dat['R'], dat['H'], dat['L'] = R(2), H(2, 2), L
        for date in dat.index:
            if dat.loc[date, 'R'] < dat.loc[date-datetime.timedelta(9):date, 'R'].quantile(0.4):
                if dat.loc[date, 'L'] > dat.loc[date, 'R'] and dat.loc[date, 'R'] > dat.loc[date, 'H']:
                    dat.loc[date, 'RHL_'] = 1
            elif dat.loc[date, 'R'] > dat.loc[date-datetime.timedelta(9):date, 'R'].quantile(0.6):
                if dat.loc[date, 'L'] < dat.loc[date, 'R'] or dat.loc[date, 'R'] < dat.loc[date, 'H']:
                    dat.loc[date, 'RHL_'] = -1
        return dat

    # def KDJ(self, dat, n=9, m1=3, m2=3):
    #     k = self.EMA(dat['PX_LAST'], 2)
    #     d = self.EMA(k, 2)
    #     j = 3*k - 2*d
    #     return k, d, j
    # def KDJ_(self, dat, n=9, m1=3, m2=3):
    #     dat['KDJ_K'], dat['KDJ_D'], dat['KDJ_J'] = self.KDJ(dat, n, m1, m2)
    #     dat['KDJ_'] = 0
    #     # for date in dat.index:
    #     #     if dat.loc[date, 'KDJ_K'] < dat['KDJ_K'].quantile(0.4):
    #     #         if dat.loc[date, 'KDJ_J'] > dat.loc[date, 'KDJ_K']\
    #     #             and dat.loc[date, 'KDJ_K'] > dat.loc[date, 'KDJ_D']:
    #     #             dat.loc[date, 'KDJ_'] = 1
    #     # for date in dat.index:
    #     #     if dat.loc[date, 'KDJ_K'] > dat['KDJ_K'].quantile(0.6):
    #     #         if dat.loc[date, 'KDJ_J'] < dat.loc[date, 'KDJ_K'] \
    #     #                 or dat.loc[date, 'KDJ_K'] < dat.loc[date, 'KDJ_D']:
    #     #             dat.loc[date, 'KDJ_'] = -1
    #     for date in dat.index:
    #         if dat.loc[date, 'KDJ_K'] < dat.loc[date - datetime.timedelta(n):date, 'KDJ_K'].quantile(0.4):
    #             if dat.loc[date, 'KDJ_J'] > dat.loc[date, 'KDJ_K'] \
    #                                 and dat.loc[date, 'KDJ_K'] > dat.loc[date, 'KDJ_D']:
    #                 dat.loc[date, 'KDJ_'] = 1
    #         elif dat.loc[date, 'KDJ_K'] > dat.loc[date - datetime.timedelta(n):date, 'KDJ_K'].quantile(0.6):
    #             if dat.loc[date, 'KDJ_J'] < dat.loc[date, 'KDJ_K'] \
    #                                 or dat.loc[date, 'KDJ_K'] < dat.loc[date, 'KDJ_D']:
    #                 dat.loc[date, 'KDJ_'] = -1
    #     return dat


    def RSI(self, dat, n=6):
        temp = dat['PX_LAST']/dat['PX_LAST'].shift(1) - 1
        tmp = temp.copy()
        tmp[tmp < 0] = 0
        rsi = sum(tmp.shift(i) for i in range(n)) / sum(abs(temp.shift(i)) for i in range(n)) * 100
        return rsi
    def RSI_(self, dat, n=6):
        dat['RSI'] = self.RSI(dat, n)
        dat['RSI_'] = 0
        # self.interval(dat['RSI'], 50, 80)
        # dat.loc[self.interval(dat['RSI'], 50, 80), 'RSI_'] = 1
        # dat.loc[self.interval(dat['RSI'], 80, 100), 'RSI_'] = -1
        # dat.loc[self.interval(dat['RSI'], 0, 20), 'RSI_'] = 1
        for date in dat.index:
            if self.divergence(dat, 5, date, 'RSI') == -1 \
                and (dat.loc[date, 'RSI'] <= 20
                or  (dat.loc[date, 'RSI'] > 50 and dat.loc[date, 'RSI'] <= 80)):
                dat.loc[date, 'RSI_'] = 1
            elif self.divergence(dat, 5, date, 'RSI') == 1 \
                or dat.loc[date, 'RSI'] > 80:
                dat.loc[date, 'RSI_'] = -1
        return dat

    def RSI__(self, dat, n=6):
        dat['RSI'] = self.RSI(dat, n)
        dat['RSI_'] = 0
        dat.loc[dat['RSI'] > 80, 'RSI_'] = 1
        dat.loc[dat['RSI'] < 20, 'RSI_'] = -1
        return dat

    def CMO(self, dat, n=12):
        temp = dat['PX_LAST'] / dat['PX_LAST'].shift(1) - 1
        tmp = temp.copy()
        tmp[tmp < 0] = 0
        tp = temp.copy()
        tp[tp > 0] = 0
        cmo = sum(tmp.shift(i)+tp.shift(i) for i in range(n)) / sum(abs(temp.shift(i)) for i in range(n)) * 100
        return cmo
    def CMO_(self, dat, n=12):
        dat['CMO'] = self.CMO(dat, n)
        dat['CMO_'] = 0
        dat.loc[dat['CMO'] > 0, 'CMO_'] = 1
        dat.loc[dat['CMO'] < 0, 'CMO_'] = -1
        return dat

    def attitude(self, dat, n, factor):
        dat[factor+'_A'] = 0
        for i, date in enumerate(dat.index):
            if 1 in list(dat.loc[dat.index[i-n]:date, factor+'_']):
                dat.loc[date, factor+'_A'] += 1
            if -1 in list(dat.loc[dat.index[i-n]:date, factor+'_']):
                dat.loc[date, factor+'_A'] -= 1
        return dat
    def KING(self, dat, king_list=['ROC', 'CCI', 'MACD', 'RSI', 'RHL']):
        dat['KING'] = 0
        weight = [1,1,0.5,1,1.5]
        for factor in king_list:
            dat = eval('self.%s_(dat)' % factor)
            dat = eval('self.attitude(dat, 2, \'%s\')' % factor)
            if factor == 'RHL':
                dat['KING'] += dat['%s_A' % factor]*2
            elif factor == 'MACD':
                dat['KING'] += dat['%s_A' % factor]*0.5
            else:
                dat['KING'] += dat['%s_A' % factor]
        # dat = self.ROC_(dat)
        # dat = self.CCI_(dat)
        # dat = self.MACD_(dat)
        # dat = self.RSI_(dat)
        # dat = self.RHL_(dat)
        return dat
    def KING_(self, dat, king_list=['ROC', 'CCI', 'MACD', 'RSI', 'RHL']):
        dat = self.KING(dat, king_list)
        dat['KING_'] = 0
        dat.loc[dat['KING'] >= 2, 'KING_'] = 1
        dat.loc[dat['KING'] <=-2, 'KING_'] = -1
        return dat

    def win_rate(self, dat, factor):
        # dat = self.evaluate(dat, factor)
        price = dat['PX_LAST'][0]
        hold = 0
        win = 0
        count = 0
        for date in dat.index:
            if dat.loc[date, factor+'_'] == 1:
                dat.loc[date, 'positive'] = 1
                if hold == 0:
                    price = dat.loc[date, 'PX_LAST']
                    dat.loc[date, 'buy'] = 1
                    hold = 1
                    count += 1
            elif dat.loc[date, factor+'_'] == -1:
                dat.loc[date, 'negative'] = 1
                if hold == 1:
                    if dat.loc[date, 'PX_LAST'] > price:
                        win += 1
                    dat.loc[date, 'sell'] = 1
                    hold = 0
        if hold == 1:
            dat.loc[dat.index[-1], 'sell'] = min(dat['PX_LOW'])
            dat.loc[dat.index[-1], 'negative'] = min(dat['PX_LOW'])
            if dat['PX_LAST'][-1] > price:
                win += 1
        if count == 0:
            print('No trade by %s' % factor)
            dat.loc[dat.index[0], 'buy'] = min(dat['PX_LOW'])
            dat.loc[dat.index[0], 'sell'] = min(dat['PX_LOW'])
            dat.loc[dat.index[0], 'positive'] = min(dat['PX_LOW'])
            dat.loc[dat.index[0], 'negative'] = min(dat['PX_LOW'])
            # dat['buy'] = 0
            # dat['sell'] = 0
            # dat['positive'] = 0
            # dat['negative'] = 0
        else:
            print('Win rate of %s: ' % factor, win / count, ' trade %s times' % count)
        return dat

    def factor_plot(self, dat, factor):
        dat = dat.loc[:, ['ts_code', 'PX_LAST', 'PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_VOLUME']]
        dat = self.evaluate(dat, [factor])
        dat = self.win_rate(dat, factor)
        temp = dat.copy()
        temp.rename(columns={'PX_OPEN':'Open', 'PX_HIGH':'High', 'PX_LOW':'Low',
                             'PX_LAST':'Close', 'PX_VOLUME':'Volume'}, inplace=True)
        temp.loc[temp['buy']==1, 'buy'] = temp['Low']*0.98
        temp.loc[temp['sell']==1, 'sell'] = temp['High']*1.02
        temp.loc[temp['positive']==1, 'positive'] = temp['Low']*0.98
        temp.loc[temp['negative']==1, 'negative'] = temp['High']*1.02
        # temp.loc[temp['buy'] == 1, 'buy'] = temp['Low'] * 0.999
        # temp.loc[temp['sell'] == 1, 'sell'] = temp['High'] * 1.001
        # temp.loc[temp['positive'] == 1, 'positive'] = temp['Low'] * 0.999
        # temp.loc[temp['negative'] == 1, 'negative'] = temp['High'] * 1.001
        try:
            apds = [mpf.make_addplot(temp[factor], panel=1, color='g', alpha=0.3),
                    mpf.make_addplot(temp['positive'], type='scatter', markersize=70, marker='^', color='r', alpha=0.4),
                    mpf.make_addplot(temp['negative'], type='scatter', markersize=70, marker='v', color='green', alpha=0.4),
                    mpf.make_addplot(temp['buy'], type='scatter', markersize=100, marker='^', color='red'),
                    mpf.make_addplot(temp['sell'], type='scatter', markersize=100, marker='v', color='green', title=factor)]
            mpf.plot(temp, type='candle', addplot=apds, volume=True,
                     datetime_format='%Y-%m-%d', xrotation=45, figscale=1.0)
        except:
            apds = [mpf.make_addplot(temp['positive'], type='scatter', markersize=70, marker='^', color='r', alpha=0.4),
                    mpf.make_addplot(temp['negative'], type='scatter', markersize=70, marker='v', color='green', alpha=0.4),
                    mpf.make_addplot(temp['buy'], type='scatter', markersize=100, marker='^', color='red'),
                    mpf.make_addplot(temp['sell'], type='scatter', markersize=100, marker='v', color='green', title=factor)]
            mpf.plot(temp, type='candle', addplot=apds, volume=True,
                     datetime_format='%Y-%m-%d', xrotation=45, figscale=1.0)
        plt.show()

        # grid = plt.GridSpec(3,4,wspace=0.5,hspace=0.5)
        # # ax1 = plt.subplot(grid[0:2, 0:4])
        # fig = mpf.figure(figsize=(6,6))
        # ax1 = fig.add_subplot(grid[0:3, 0:4])
        # ax1.plot(temp, type='candle', volume=True)
        # # ax1.plot(dat['PX_LAST'], 'r')
        # # ax1.set_title('Close Price')
        # ax2 = fig.add_subplot(grid[2:3, 0:4])
        # ax2.plot(temp['score'], alpha=0.3)
        # ax3 = ax2.twinx()
        # try:
        #     ax3.plot(temp[factor], label=factor)
        # except:
        #     pass
        #     # print("%s can't plot" % factor)
        # plt.legend()
        # plt.show()

    def evaluate(self, dat, factor_list):
        dat['return'] = dat['PX_LAST'] / dat['PX_LAST'].shift(1) - 1
        dat['score'] = 0
        for factor in factor_list:
            dat = eval('self.%s_(dat)' % factor)
            dat.loc[:, 'score'] = dat.loc[:, 'score'] + dat.loc[:, factor+'_']
        return dat
    def evaluate_plot(self, dat, factor_list):
        dat = self.evaluate(dat, factor_list)
        # grangercausalitytests(dat[['return', 'score']].dropna(), maxlag=2)
        # print('correlation between return and score: ', dat[['return', 'score']].corr().iloc[0, 1])
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(dat['PX_LAST'], label='close', color='red')
        ax2 = ax1.twinx()
        ax2.plot(dat['score'].fillna(0), label='score', color='green', alpha=0.3)
        plt.legend()
        plt.show()