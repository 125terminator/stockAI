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
        self.macd_signal_cross = []
        self.inputs = []
        self.prepare_inputs()

    def prepare_inputs(self):
        self.compute_moving_averages()
        self.compute_rsi()
        self.compute_dema()
        self.compute_tema()
        self.compute_stoch()
        self.compute_macd()
        self.inputs = np.concatenate((self.moving_averages, \
            self.rsi_list, self.dema_list, self.tema_list, self.stoch_list,\
            self.macd_signal_cross), axis=0)

    # Exponential moving average
    def ema(self, series, n):
        return series.ewm(span=n, min_periods=n).mean()

    def macd(self, n):
        exp1 = ema(self.df.Close, 12*n)
        exp2 = ema(self.df.Close, 26*n)
        macd = exp1-exp2
        exp3 = ema(macd, 9*n)
        return np.array(2*(macd>exp3).shift(1)-1)

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

    def compute_macd(self):
        periods = [1, 10, 50, 100]
        for period in periods:
            self.macd_signal_cross.append(self.macd(period))

        self.macd_signal_cross = np.array(self.macd_signal_cross)

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