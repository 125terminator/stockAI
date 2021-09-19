from scipy.stats import zscore

class Inputs:
    def __init__(self, df):
        self.df = df
        self.moving_averages = []
        self.inputs = []
        self.prepare_inputs()

    def prepare_inputs(self):
        self.get_moving_averages()
        self.inputs = self.moving_averages
    
    def get_moving_averages(self):
        periods = [1, 3, 5, 7, 11, 15, 19, 23, 27, 35, 41, 50, 61, 128, 375]
        moving_averages = []
        for period in periods:
            m = self.df.Close.rolling(period).mean()
            moving_averages.append(m)
        
        self.moving_averages = np.array(moving_averages)
        for i in range(375, self.moving_averages.shape[1]):
            self.moving_averages[:, i] = self.moving_averages[0, i] - self.moving_averages[:, i]

        self.moving_averages = np.delete(self.moving_averages, 0, 0)