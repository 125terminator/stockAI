from datetime import datetime
from dateutil.parser import parse
from random import randrange

import numpy as np

from broker_calc import get_net_profit


class Robo:
    def __init__(self, ann, df, money, inputs, logger):
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
        self.logger = logger
        self.logger.write("Start of stock buying {}\n".format(self))

    def sell_stocks(self, i):
        if self.bought[0] == True:
            sell_price = self.df.Close[i]
            buy_price = self.bought[2]
            stock_qty = self.bought[1]

            net_profit = get_net_profit(buy_price, sell_price, stock_qty)
            log = "{time} Sold at = {sold:.2f}\t Profit = {profit:.2f} " \
                "{tot_profit:.2f} {bp:.2f} {sp:.2f} {qty}\n" \
                .format(time=self.df.Time[i], sold=buy_price*stock_qty + net_profit, \
                profit=net_profit, tot_profit=self.profit + self.loss + net_profit, \
                bp=buy_price, sp=sell_price, qty=stock_qty)
            self.logger.write(log)

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
        start_index = 375*10
        end_index = 10088*2+37
        output = self.ann.forward_propagation(self.inputs)
        # output index meaning
        # 0 -> buy if money
        # 1 -> sell if has stocks
        # 2 -> hold

        for i in range(start_index, end_index):
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
                        log = "{time} Bought at = {bought:.2f} \t".format(\
                            time=self.df.Time[i], bought=stock_qty*bought_price)
                        self.logger.write(log)

                elif ind == 1:
                    self.sell_stocks(i)
                    
        self.logger.write("\nEnd of stock buying Total profit = {}\n\n\n".format(self.profit + self.loss))
        self.logger.flush()
        return self.profit + self.loss 