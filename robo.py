from io import DEFAULT_BUFFER_SIZE
from scipy.stats import zscore
from broker_calc import get_net_profit
from datetime import datetime


class Robo:
    def __init__(self, ann, df, money):
        self.ann = ann
        self.df = df
        self.money = money
        self.init_money = money
        # self.bought -> [has stocks, number of stocks, price of each stock]
        self.bought = [False, 0, 0]
        self.moving_averages = []
        self.time_ind = []
    
    def fitness(self):
        self.prepare_inputs()
        output = self.ann.forward_propagation(self.moving_averages)
        # 0 -> buy if money
        # 1 -> sell if has stocks
        # 2 -> hold
        start_date_index = 60
        while df.Date[start_date_index][12] != 9 or df.Date[start_date_index][14:16]!=15:
            start_date_index += 1

        for i in range(start_date_index, output.shape[1]):
            ind = output[:, i].argmax()

            if ind == 0:
                bought_price = self.df.Close[i]
                if self.money >= bought_price:
                    stock_qty = self.money // bought_price
                    self.bought = [True, stock_qty, bought_price]
                    self.money = self.money - stock_qty * bought_price

            elif ind == 1:
                if self.bought[0] == True:
                    sell_price = self.df.Close[i]
                    buy_price = self.bought[2]
                    stock_qty = self.bought[1]

                    holdings = buy_price*stock_qty + get_net_profit(buy_price, sell_price, stock_qty)
                    self.money += holdings
                    self.bought = [False, 0, 0]

            
            if df.Date[i][11:13]==15 and df.Date[i][14:16]==29:
                self.bought = [False, 0, 0]

        return self.money

    def prepare_inputs(self):
        self.get_moving_averages()

    def get_time_as_absindex(self):
        # return 09:15 as 1, 09:16 as 2, so on
        init_time = datetime.fromisoformat('2015-03-02T09:14:00')
        time = []
        for i in range(len(df)):
            ind = datetime.fromisoformat(df.Date[i]) - init_time
            time.append(ind.seconds // 60)
        
        self.time_ind = np.array(time)

    def get_moving_averages(self):
        periods = [1, 3, 5, 7, 11, 15, 19, 23, 27, 35, 41, 50, 61, 128]
        moving_averages = []
        for period in periods:
            m = self.df.Close.rolling(period).mean()
            moving_averages.append(m)
        self.moving_averages = zscore(moving_averages)