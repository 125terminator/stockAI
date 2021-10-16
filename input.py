import numpy as np
import pandas as pd
from scipy.stats import zscore
import math

class Inputs:
    def __init__(self, df):
        self.df = df
        self.moving_averages = []
        self.rsi_list = []
        self.dema_list = []
        self.tema_list = []
        self.stoch_list = []
        self.vin_list = []
        self.vip_list = []
        self.cci_list = []
        self.avg_true_list = []
        self.macd_signal_cross = []
        self.obv_cross_list = []
        self.inputs = []
        self.prepare_inputs()

        self.first_non_nan = 0
        for i in range(self.inputs.shape[0]):
            for j in range(self.inputs.shape[1]):
                if math.isnan(self.inputs[i, j]):
                    self.first_non_nan = max(self.first_non_nan, j)

        self.start_day = 0
        for i in range(self.first_non_nan, self.df.shape[0]):
            self.start_day = i
            if self.df.Date[i] == 1:
                break

        self.end_day = self.start_day + 375*60
        print('start day: ', self.start_day)
        print('end day: ', self.end_day)

    def prepare_inputs(self):
        self.compute_moving_averages()
        self.compute_rsi()
        self.compute_dema()
        self.compute_tema()
        self.compute_stoch()
        self.compute_vortex_indicator_pos()
        self.compute_vortex_indicator_neg()
        self.compute_cci()
        self.compute_macd()
        self.compute_obv()
        self.compute_average_true_range()
        self.inputs = np.concatenate((self.moving_averages, \
            self.rsi_list, self.dema_list, self.tema_list, \
            self.stoch_list, self.vip_list, self.vin_list, self.cci_list,\
            self.avg_true_list, self.macd_signal_cross, self.obv_cross_list), axis=0)

    # Exponential moving average
    def ema(self, series, n):
        return series.ewm(span=n, min_periods=n, ignore_na=True).mean()

    def macd(self, n):
        exp1 = self.ema(self.df.Close, 12*n)
        exp2 = self.ema(self.df.Close, 26*n)
        macd = exp1-exp2
        exp3 = self.ema(macd, 9*n)
        return np.array(2*(macd>exp3)-1)

    # Double exponential average
    def DEMA(self, n):
        EMA = self.ema(self.df.Close, n)
        return 2*EMA - self.ema(EMA,n)

    # Triple exponential average
    def TEMA(self, n):
        EMA = self.ema(self.df.Close, n)
        EEMA = self.ema(EMA, n)
        return 3*EMA - 3*EEMA + self.ema(EEMA, n)

    def stoch(self, n=14):
        smin = self.df.Low.rolling(n).min()
        smax = self.df.High.rolling(n).max()
        stoch_k = 100 * (self.df.Close - smin) / (smax - smin)
        return np.array(stoch_k)

    def stoch_signal(self, n=14, d_n=3):
        stoch_k = pd.Series(self.stoch(n), name='stoch_k')
        stoch_d = stoch_k.rolling(d_n).mean()
        return np.array(stoch_d)

    def rsi(self, n=14):
        diff = self.df.Close.diff(1)
        which_dn = diff < 0
        up, dn = diff, diff*0
        up[which_dn], dn[which_dn] = 0, -up[which_dn]
        emaup = self.ema(up, n)
        emadn = self.ema(dn, n)
        rsi = 100 * emaup / (emaup + emadn)
        return np.array(rsi)

    def average_true_range(self, n=14):
        cs = self.df.Close.shift(1)
        tr = self.df.High.combine(cs, max) - self.df.Low.combine(cs, min)

        atr = np.zeros(len(self.df.Close))
        atr[0] = tr[1::].mean()
        for i in range(1, len(atr)):
            atr[i] = (atr[i-1] * (n-1) + tr.iloc[i]) / float(n)

        return np.array(atr)

    def vortex_indicator_pos(self, n=14):
        tr = self.df.High.combine(self.df.Close.shift(1), max) - self.df.Low.combine(self.df.Close.shift(1), min)
        trn = tr.rolling(n).sum()

        vmp = np.abs(self.df.High - self.df.Low.shift(1))
        vmm = np.abs(self.df.Low - self.df.High.shift(1))

        vip = vmp.rolling(n).sum() / trn
        return np.array(vip)

    def vortex_indicator_neg(self, n=14):
        tr = self.df.High.combine(self.df.Close.shift(1), max) - self.df.Low.combine(self.df.Close.shift(1), min)
        trn = tr.rolling(n).sum()

        vmp = np.abs(self.df.High - self.df.Low.shift(1))
        vmm = np.abs(self.df.Low - self.df.High.shift(1))

        vin = vmm.rolling(n).sum() / trn
        return np.array(vin)

    def cci(self, n=20, c=0.015):
        pp = (self.df.High + self.df.Low + self.df.Close) / 3.0
        cci = (pp - pp.rolling(n).mean()) / (c * pp.rolling(n).std())
        return np.array(cci)

    def compute_macd(self):
        periods = [1, 10, 50, 100]
        for period in periods:
            self.macd_signal_cross.append(self.macd(period))

        self.macd_signal_cross = np.array(self.macd_signal_cross)

    def compute_obv(self):
        obv = (np.sign(self.df.Close.diff()) * self.df.Volume).fillna(0).cumsum()
        self.obv_cross_list.append(np.array(2*(obv>obv.shift(1))-1))
        self.obv_cross_list = np.array(self.obv_cross_list)

    def compute_average_true_range(self):
        periods = [375*10]
        for period in periods:
            m = self.average_true_range(period)
            self.avg_true_list.append(m)

        self.avg_true_list = zscore(self.avg_true_list, axis=1, nan_policy='omit')

    def compute_vortex_indicator_pos(self):
        periods = [30, 60, 180, 375, 375*5, 375*10]
        for period in periods:
            m = self.vortex_indicator_pos(period)
            self.vip_list.append(m)

        self.vip_list = zscore(self.vip_list, axis=1, nan_policy='omit')

    def compute_vortex_indicator_neg(self):
        periods = [30, 60, 180, 375, 375*5, 375*10]
        for period in periods:
            m = self.vortex_indicator_neg(period)
            self.vin_list.append(m)

        self.vin_list = zscore(self.vin_list, axis=1, nan_policy='omit')

    def compute_cci(self):
        periods = [30, 60, 180, 375, 375*5, 375*10]
        for period in periods:
            m = self.cci(period)
            self.cci_list.append(m)

        self.cci_list = zscore(self.cci_list, axis=1, nan_policy='omit')

    def compute_stoch(self):
        periods = [30, 60, 180, 375, 375*5, 375*10]
        for period in periods:
            m = self.stoch(period)
            self.stoch_list.append(m)

        self.stoch_list.append(self.stoch_signal(375*10, 375*4))
        self.stoch_list = zscore(self.stoch_list, axis=1, nan_policy='omit')

    def compute_tema(self):
        periods = [30, 60, 180, 375, 375*5, 375*10]
        for period in periods:
            m = self.TEMA(period)
            self.tema_list.append(m)

        self.tema_list =  zscore(self.tema_list, axis=1, nan_policy='omit')

    def compute_dema(self):
        periods = [30, 60, 180, 375, 375*5, 375*10]
        for period in periods:
            m = self.DEMA(period)
            self.dema_list.append(m)

        self.dema_list =  zscore(self.dema_list, axis=1, nan_policy='omit')
        
    def compute_rsi(self):
        periods = [30, 60, 180, 375, 375*5, 375*10]
        for period in periods:
            m = self.rsi(period)
            self.rsi_list.append(m)
        
        self.rsi_list = np.array(self.rsi_list)
        self.rsi_list /= 100
    
    def compute_moving_averages(self):
        periods = [1, 30, 60, 180, 375, 375*5, 375*10]
        moving_averages = []
        for period in periods:
            m = self.df.Close.rolling(period).mean()
            moving_averages.append(m)
        
        self.moving_averages = np.array(moving_averages)
        for i in range(self.moving_averages.shape[1]):
            self.moving_averages[:, i] = self.moving_averages[0, i] - self.moving_averages[:, i]

        self.moving_averages = np.delete(self.moving_averages, 0, 0)
        self.moving_averages = zscore(self.moving_averages, axis=1, nan_policy='omit')


if __name__ == "__main__1":
    df = pd.read_csv('./data/reliance.csv')
    # 375 ind is start of 9:15 and 10088 time is 15:29
    n = 50000
    df = df[:n]
    df.index = np.arange(0, df.shape[0])

    inp = Inputs(df)