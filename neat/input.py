import numpy as np

class Inputs:
    def __init__(self, df):
        self.df = df
        self.moving_averages = []
        self.rsi_list = []
        self.inputs = []
        self.prepare_inputs()

    def prepare_inputs(self):
        self.compute_moving_averages()
        self.compute_rsi()
        self.inputs = np.concatenate((self.moving_averages, self.rsi_list), axis=0)

    def ema(self, series, n):
        return series.ewm(span=n, min_periods=n).mean()

    def rsi(self, n=14):
        diff = self.df.Close.diff(1)
        which_dn = diff < 0
        up, dn = diff, diff*0
        up[which_dn], dn[which_dn] = 0, -up[which_dn]
        emaup = self.ema(up, n)
        emadn = self.ema(dn, n)
        rsi = 100 * emaup / (emaup + emadn)
        return np.array(rsi)

    def compute_rsi(self):
        periods = [30, 60, 180, 375, 375*5, 375*10]
        self.rsi_list = []
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