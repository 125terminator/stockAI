import numpy as np
from scipy.stats import zscore

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
        self.inputs = []
        self.prepare_inputs()

    def prepare_inputs(self):
        self.compute_moving_averages()
        self.compute_rsi()
        self.compute_dema()
        self.compute_tema()
        self.compute_stoch()
        self.compute_vortex_indicator_pos()
        self.compute_vortex_indicator_neg()
        self.compute_cci()
        self.inputs = np.concatenate((self.moving_averages, \
            self.rsi_list, self.dema_list, self.tema_list, \
            self.stoch_list, self.vip_list, self.vin_list, self.cci_list), axis=0)

    # Exponential moving average
    def ema(self, series, n):
        return series.ewm(span=n, min_periods=n).mean()

    # Double exponential average
    def DEMA(self, n):
        EMA = self.ema(self.df.Close, n)
        return 2*EMA - self.ema(EMA,n)

    # Triple exponential average
    def TEMA(self, n):
        EMA = self.ema(self.df.Close, n)
        EEMA = self.ema(EMA, n)
        return 3*EMA - 3*EEMA + self.ema(EEMA, n)

    def stoch(n=14):
        smin = self.df.Low.rolling(n, min_periods=0).min()
        smax = self.df.High.rolling(n, min_periods=0).max()
        stoch_k = 100 * (self.df.Close - smin) / (smax - smin)
        return np.array(stoch_k)

    def stoch_signal(self, n=14, d_n=3):
        stoch_k = stoch(high, low, close, n, fillna=fillna)
        stoch_d = stoch_k.rolling(d_n, min_periods=0).mean()
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

    def average_true_range(n=14):
        cs = self.df.Close.shift(1)
        tr = self.df.High.combine(cs, max) - self.df.Low.combine(cs, min)

        atr = np.zeros(len(self.df.Close))
        atr[0] = tr[1::].mean()
        for i in range(1, len(atr)):
            atr[i] = (atr[i-1] * (n-1) + tr.iloc[i]) / float(n)

        return np.array(atr)

    def vortex_indicator_pos(n=14):
        tr = self.df.High.combine(self.df.Close.shift(1), max) - self.df.Low.combine(self.df.Close.shift(1), min)
        trn = tr.rolling(n).sum()

        vmp = np.abs(self.df.High - self.df.Low.shift(1))
        vmm = np.abs(self.df.Low - self.df.High.shift(1))

        vip = vmp.rolling(n, min_periods=0).sum() / trn
        return np.array(vip)


    def vortex_indicator_neg(n=14):
        tr = self.df.High.combine(self.df.Close.shift(1), max) - self.df.Low.combine(self.df.Close.shift(1), min)
        trn = tr.rolling(n).sum()

        vmp = np.abs(self.df.High - self.df.Low.shift(1))
        vmm = np.abs(self.df.Low - self.df.High.shift(1))

        vin = vmm.rolling(n).sum() / trn
        return np.array(vin)

    def cci(n=20, c=0.015):
        pp = (self.df.High + self.df.Low + self.df.Close) / 3.0
        cci = (pp - pp.rolling(n, min_periods=0).mean()) / (c * pp.rolling(n, min_periods=0).std())
        return np.array(cci)

    def compute_vortex_indicator_pos():
        periods = [30, 60, 180, 375, 375*5, 375*10]
        for period in periods:
            m = self.vortex_indicator_pos(period)
            self.vip_list.append(m)

        self.vip_list = zscore(self.vip_list, axis=1, nan_policy='omit')

    def compute_vortex_indicator_neg():
        periods = [30, 60, 180, 375, 375*5, 375*10]
        for period in periods:
            m = self.vortex_indicator_neg(period)
            self.vin_list.append(m)

        self.vin_list = zscore(self.vin_list, axis=1, nan_policy='omit')

    def compute_cci():
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
        
        self.rsi_list = zscore(self.rsi_list, axis=1, nan_policy='omit')
    
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