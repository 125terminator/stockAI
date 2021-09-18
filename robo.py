from datetime import datetime
from dateutil.parser import parse
from random import randrange

import numpy as np

from broker_calc import get_net_profit


class Robo:
    def __init__(self, ann, df, money, inputs):
        self.ann = ann
        self.df = df
        self.money = money
        self.init_money = money
        # self.bought -> [has stocks, number of stocks, price of each stock]
        self.bought = [False, 0, 0]
        self.time_ind = []
        self.profit = 0
        self.loss = 0
        self.inputs = inputs

    def sell_stocks(self, i):
        if self.bought[0] == True:
            sell_price = self.df.Close[i]
            buy_price = self.bought[2]
            stock_qty = self.bought[1]

            net_profit = get_net_profit(buy_price, sell_price, stock_qty)

            # if we are getting profit after selling stocks then
            # the amount is not credited imediately to our margin
            # profit is credited after one day
            # if loss then our margin is decreased
            if net_profit > 0:
                self.profit += net_profit
                net_profit = 0
            else:
                self.loss += net_profit
            holdings = buy_price*stock_qty + net_profit
            self.money += holdings
            self.bought = [False, 0, 0]

    def fitness(self):
        # 375 ind is start of 9:15 and 10088 time is 15:29
        start_index = 375
        output = self.ann.forward_propagation(self.inputs)
        # output index meaning
        # 0 -> buy if money
        # 1 -> sell if has stocks
        # 2 -> hold

        for i in range(start_index, output.shape[1]):
            ind = output[:, i].argmax()

            # Do not buy stocks after 15:10
            # only sell the stock
            if self.df.Date[i] > 356:
                if ind == 1:
                    self.sell_stocks(i)
                
                # sell harshly and immediately at 15:18
                if self.df.Date[i] >= 364:
                    self.sell_stocks(i)

            else:
                # regular trading
                if ind == 0:
                    bought_price = self.df.Close[i]
                    if self.bought[0] == False and self.money >= bought_price:
                        stock_qty = self.money // bought_price
                        self.bought = [True, stock_qty, bought_price]
                        self.money = self.money - stock_qty * bought_price

                elif ind == 1:
                    self.sell_stocks(i)
                    

        return self.money + self.profit + 5*self.loss 