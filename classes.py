import numpy as np
import pandas as pd
import yfinance
import yfinance as yf
from matplotlib import pyplot as plt
from scipy.optimize import minimize


class StringValidation:
    """
    class validating of the value passed is a string
    """

    def __set_name__(self, owner, name):
        self.name = name

    def __set__(self, instance, value):

        if not isinstance(value, str):
            raise AttributeError(f"Should be a string. Value passed: {value}")

        instance.__dict__[self.name] = value

    def __get__(self, instance, owner):
        if instance is None:
            return self

        return instance.__dict__[self.name]


class LetterValidation(StringValidation):
    """
    class validating if the value passed is a string and
    only contains letter. Inherits from StringValidation and adds one criteria to the setter/validation
    """

    def __set__(self, instance, value):
        value_strip = str(value).replace(" ", "")

        if not isinstance(value, str):
            raise AttributeError(f"Should be a string. Value passed: {value}")
        if not value_strip.isalpha() :
            raise AttributeError(f"Should only contain letters. Value passed: {value}")
        instance.__dict__[self.name] = value


class Investor:
    """
    class containing information about an investor
    """

    name = LetterValidation()
    last_name = LetterValidation()

    def __init__(self, name, last_name):
        self.name = name
        self.last_name = last_name

    def __repr__(self):
        return f"{self.__class__.__name__} = (name = '{self.name}', last_name = '{self.last_name}')"


class Stock:
    """
    class containing information about one stock
    the ticker should align with YFinance
    """

    ticker = StringValidation()
    name_stock = LetterValidation()

    def __init__(self, ticker, name_stock, amount, start, end):
        self.ticker = ticker
        self.name_stock = name_stock
        self.amount = amount
        self.start = start
        self.end = end
        self.data = yf.download(self.ticker, self.start, self.end, interval = "1d")[['Adj Close', 'Volume']]
        # Calculate daily returns
        self.data["DailyReturn"] = self.data["Adj Close"].pct_change()
        self.shares = self.amount/self.data["Adj Close"][0]

    def __repr__(self):
        return f"{self.__class__.__name__} = (ticker = '{self.ticker}', name_stock = '{self.name_stock}'," \
                f" amount = '{self.amount}', start = '{self.end}', start = '{self.end}')"

    def graph(self):
        self.data["Adj Close"].plot(title = f"Time Series {self.ticker}", xlabel = "Date", ylabel = "Adjusted Close Price")
        plt.savefig(f"{self.ticker} Adjusted Close Price.png")
        plt.show()

    def returns(self, daily_returns, yearly = True):
        daily_returns = round(100 * daily_returns.mean(),3)
        yearly_returns = round(daily_returns * 252, 3)

        if yearly:
            return yearly_returns

        return daily_returns

    def roi(self, adj_close):
        """
        ROI in percent
        """
        roi = (adj_close[-1]-adj_close[0])/(adj_close[0])
        roi = round(100 * roi, 3)
        return roi

    def profit(self, amount,  adj_close):
        """
        returns the (monetary) profit
        """
        return amount * self.roi(adj_close = adj_close)/100

    def risk(self, daily_return, yearly=True):
        """
        returns the risk measured as the standard deviation of the returns
        yearly or daily returns
        """
        daily_std = round(100 * daily_return.std(), 3)
        yearly_std = round(daily_std * np.sqrt(252), 3)

        if yearly:
            return yearly_std

        return daily_std

    def sharpe_ratio(self, daily_return, risk_free_rate_yearly = 0):
        """
        returns the Sharpe Ratio expressed as:
        (return-risk_free_rate)/std
        """
        volatility = self.risk(daily_return, yearly=True)
        return_ = self.returns(daily_return, yearly=True)
        sharpe_ratio = (return_- risk_free_rate_yearly)/volatility
        return sharpe_ratio

    def performance_report(self,adj_close, daily_return, export_report = False):
        """
        returns a performance report with the below KPIs
        """

        data = {}
        data["Ticker"] = self.ticker
        data["Amount"] = self.amount
        data["Shares"] = self.shares
        data["Period begin"] = self.start
        data["Period end"] = self.end
        data["Mean daily return %"] = self.returns(daily_return, yearly = False)
        data["Mean yearly return %"] = self.returns(daily_return, yearly = True)
        data["ROI"] = self.roi(adj_close)
        data["Profit"] = self.profit(self.amount, adj_close)
        data["Daily Risk"] = self.risk(daily_return, yearly = False)
        data["Yearly Risk"] = self.risk(daily_return, yearly = True)
        data["Sharpe Ratio"] = self.sharpe_ratio(daily_return, risk_free_rate_yearly = 0)
        df = pd.DataFrame.from_dict(data, orient = 'index')
        df = df.rename(columns = {0:""})
        print("Returning performance report")

        if export_report:
            filename = f"{self.ticker}_PerformanceReport v2.xlsx"
            df.to_excel(filename)

        return df

    def daily_returns_table(self):
        df = self.data.copy()
        df[f"DailyReturn"] = df["Adj Close"].pct_change()
        df = df.dropna(axis = 0)
        return df

    def add_suffix_on_columns(self):
        """
        adds the suffix defined by the stocks ticker to
        the existing columns of the data attribute
        """
        return self.data.add_suffix(f"_{self.ticker}")


class Portfolio(Stock):

    """
    class containing a set of stock and building a portfolio out of them
    """

    def __init__(self, stocks, start, end, benchmark_ticker = "^GSPC"):
        self.stocks = stocks
        self.start = start
        self.end = end
        self.benchmark_ticker = benchmark_ticker

        self.total_amount = sum([stock.amount for stock in self.stocks])

    def benchmark_data(self):
        """
        returns both the daily return and the Adj Close for the
        chosen Benchmark
        """
        df = yfinance.download(tickers = self.benchmark_ticker, start = self.start, end = self.end, interval = "1d")[['Adj Close']]
        df['DailyReturn'] = df["Adj Close"].pct_change()
        df = df.add_suffix(f"_{self.benchmark_ticker}")
        df = df.dropna(axis = 0)
        return df

    def stock_data(self):
        """
        concatenates Adjusted Close and DailyReturn
        from all Stock instances passed
        """

        df = [s.add_suffix_on_columns() for s in self.stocks]
        df = pd.concat(df, axis=1)

        cols = df.columns
        cols = [columns for columns in cols if "DailyReturn" in columns or "Adj Close" in columns]

        df = df[cols].dropna(axis = 0)

        return df

    def portfolio_data(self):
        """
        Creates a df containing an index build with all Stock-instances passed
        index = sum(weight*adj_close) where weight is the percentage of the amount allocated to that stock
        as well as the daily returns
        """
        portfolio_index = sum([(stock.amount/self.total_amount) * stock.data["Adj Close"] for stock in self.stocks])

        portfolio_index = portfolio_index.to_frame()
        portfolio_index["DailyReturn"] = portfolio_index["Adj Close"].pct_change()
        portfolio_index = portfolio_index.dropna(axis = 0)
        return portfolio_index

    def beta(self):
        """
        calculates the Beta coefficient
        portfolio vs benchmark
        """
        # retrieve all data and filter
        index_data = self.portfolio_data()
        benchmark_data = self.benchmark_data()
        data = pd.concat([index_data, benchmark_data], axis = 1)
        data = data.dropna(axis = 0)
        cols = [columns for columns in data.columns if "DailyReturn" in columns]
        data = data[cols]

        print(data.columns)

        market_variance = data[f"DailyReturn_{self.benchmark_ticker}"].var() * 252
        cov = data.cov() * 252
        beta = cov.iloc[0, 1] / market_variance

        return beta


    def all_data(self, export = False):
        """
        Creates a df with Adj Close and DailyReturn info for the Portfolio,
         Stocks and Benchmark
        """
        stock_data = self.stock_data()
        index_data = self.portfolio_data()
        benchmark_data = self.benchmark_data()
        all_data = pd.concat([index_data, stock_data, benchmark_data], axis = 1)
        all_data = all_data.dropna(axis = 0)

        if export:
            all_data.to_excel("All_data.xlsx")

        return all_data

    def normalized_graph(self):
        """
        Returns a linechart with normalized Adj Close Prices:
        """
        # retrieve all data and filter
        df = self.all_data()
        cols = df.columns
        cols = [columns for columns in cols if "Adj Close" in columns]
        df = df[cols]

        # normalize to 100
        df = 100 * df/df.iloc[0]

        df.plot(title = "Normalized Close Prices", xlabel = "Date", ylabel = "Adjusted Close Price")
        plt.savefig('ClosingPrices.png', bbox_inches='tight')

    @staticmethod
    def gen_weights(N):
        weights = np.random.random(N)
        return weights / np.sum(weights)

    @staticmethod
    def calculate_returns(weights, log_rets):
        return np.sum(log_rets.mean() * weights) * 252  # Annualized Log-Return

    @staticmethod
    def calculate_volatility(weights, log_rets_cov):
        annualized_cov = np.dot(log_rets_cov * 252, weights)
        vol = np.dot(weights.transpose(), annualized_cov)
        return np.sqrt(vol)

    def function_to_minimize(self, weights):
        """
        minimize the sharpe ratio: return/risk
        """
        # Note -1* because we need to minimize this
        # Its the same as maximizing the positive sharpe ratio

        # retrieve all data and filter
        df = self.stock_data()
        cols = df.columns
        cols = [columns for columns in cols if "Adj Close_" in columns]  # leave Portfolio data out
        df = df[cols]

        # log returns and covariance
        log_rets = np.log(df / df.shift(1))
        log_rets_cov = log_rets.cov()

        return -1 * (self.calculate_returns(weights, log_rets) / self.calculate_volatility(weights, log_rets_cov))

    def markowitz_frontier_graph(self):
        """
        returns optimal portfolio weights in terms of Sharpe Ratio as
        well as a png with MC Simulation for 6000 random weights
        and their risk/return
        """

        # retrieve all data and filter
        df = self.stock_data()
        cols = df.columns
        cols = [columns for columns in cols if "Adj Close_" in columns] # leave Portfolio data out
        df = df[cols]

        # log returns and covariance
        log_rets = np.log(df / df.shift(1))
        log_rets_cov = log_rets.cov()

        # Now we just create many, many random weightings and
        # we can then plot them on expected return vs. expected volatility (coloring them by Sharpe Ratio):
        # Monte Carlo Simulation
        mc_portfolio_returns = []
        mc_portfolio_vol = []
        mc_weights = []
        N = len(df.columns)
        for sim in range(6000):
            # This may take a while!
            weights = self.gen_weights(N)
            mc_weights.append(weights)
            mc_portfolio_returns.append(self.calculate_returns(weights, log_rets))
            mc_portfolio_vol.append(self.calculate_volatility(weights, log_rets_cov))


        mc_sharpe_ratios = np.array(mc_portfolio_returns) / np.array(mc_portfolio_vol)



        # 18.3 CREATE a df

        mc_weights_rd = [np.round(weight, 3) for weight in mc_weights]
        mc_portfolio_returns_rd = [round(returns,3) for returns in mc_portfolio_returns]
        mc_portfolio_vol_rd = [round(vol,3) for vol in mc_portfolio_vol]
        mc_sharpe_ratios_rd = np.round(mc_sharpe_ratios, 3)

        data = {"Weights" : mc_weights_rd,
                "Return" : mc_portfolio_returns_rd,
                "Risk" : mc_portfolio_vol_rd,
                "Sharpe Ratio" : mc_sharpe_ratios_rd}


        sharpe_df = pd.DataFrame(data = data).sort_values(by='Sharpe Ratio',ascending=False)

        plt.figure(dpi=100, figsize=(10, 5))
        plt.scatter(mc_portfolio_vol, mc_portfolio_returns, c=mc_sharpe_ratios)
        plt.ylabel('Expected Returns')
        plt.xlabel('Expected Volatility')
        plt.colorbar(label="Sharpe Ratio")
        plt.savefig('MC_Sharpe.png', bbox_inches='tight')
        plt.show()

        # Optimal Weighting through Minimization Search

        bounds = tuple((0,1) for n in range(N))

        # Starting Guess
        equal_weights = N * [1/N]

        # Need to constrain all weights to add up to 1
        sum_constraint = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

        # scipy minimize module
        res = minimize(fun = self.function_to_minimize, x0=equal_weights, bounds=bounds, constraints=sum_constraint)

        return {"Optimal portfolio weights" : res.x,
                "Sharpe Ratio": (-1* res.fun),
                "MC_Simulations" : sharpe_df}


    def graph_returns(self):
        """
        returns a histogram with daily returns
        """
        df = self.all_data()
        cols = df.columns
        cols = [columns for columns in cols if "DailyReturn" in columns]
        df = df[cols]

        df.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);
        plt.savefig('ReturnsHistogram.png', bbox_inches='tight')
        plt.show()

    def performance_report(self, export_report = True):

        benchmark_data = self.benchmark_data()
        portfolio_data = self.portfolio_data()
        print(portfolio_data)

        cols_benchmark = benchmark_data.columns
        cols_portfolio = portfolio_data.columns

        cols_benchmark_daily_return = [col for col in cols_benchmark if "DailyReturn" in col]
        cols_benchmark_adj_close = [col for col in cols_benchmark if "Adj Close" in col]

        cols_portfolio_daily_return = [col for col in cols_portfolio if "DailyReturn" in col]
        cols_portfolio_adj_close = [col for col in cols_portfolio if "Adj Close" in col]

        daily_return_benchmark = benchmark_data[cols_benchmark_daily_return]
        adj_close_benchmark = benchmark_data[cols_benchmark_adj_close]

        daily_return_portfolio = portfolio_data[cols_portfolio_daily_return]
        adj_close_portfolio = portfolio_data[cols_portfolio_adj_close]

        ## First df (Portfolio data)
        data = {}
        data["Ticker"] = ','.join([str(stock.ticker) for stock in self.stocks])
        data["Amount"] = ','.join([str(stock.amount) for stock in self.stocks])
        data["Total Amount"] = self.total_amount
        data["Period begin"] = self.start
        data["Period end"] = self.end
        data["Mean daily return %"] = self.returns(daily_return_portfolio["DailyReturn"],  False)
        data["Mean yearly return %"] = self.returns(daily_return_portfolio["DailyReturn"], True)
        print(adj_close_portfolio)
        data["ROI %"] = self.roi(adj_close_portfolio["Adj Close"])
        data["Profit"] = self.profit(self.total_amount, adj_close_portfolio["Adj Close"])
        data["Daily Risk"] = self.risk(daily_return_portfolio["DailyReturn"], False)
        data["Yearly Risk"] = self.risk(daily_return_portfolio["DailyReturn"], True)
        data["Sharpe Ratio"] = round(self.sharpe_ratio(daily_return_portfolio["DailyReturn"],  0), 3)
        data["Beta"] = round(self.beta(),3)

        markowitz_results = self.markowitz_frontier_graph() # returns a dict {key1:list1, key2:number, key3:df}
        optimal_weights = [round(num, 4) for num in markowitz_results["Optimal portfolio weights"]]
        optimal_sharpe = round(markowitz_results["Sharpe Ratio"],4)
        sharpe_df = markowitz_results["MC_Simulations"]
        data["Optimal Weights"] = optimal_weights
        data["Optimal Sharpe Ratio"] = optimal_sharpe

        df = pd.DataFrame([data]).transpose()
        df = df.rename(columns = {0:"Portfolio"})

        print(df)

        ## Second df (Benchmark data)
        data = {}
        data["Ticker"] = self.benchmark_ticker
        data["Amount"] = self.total_amount
        data["Total Amount"] = self.total_amount
        data["Period begin"] = self.start
        data["Period end"] = self.end
        data["Mean daily return %"] = self.returns(daily_return_benchmark[f"DailyReturn_{self.benchmark_ticker}"],  False)
        data["Mean yearly return %"] = self.returns(daily_return_benchmark[f"DailyReturn_{self.benchmark_ticker}"], True)
        data["ROI %"] = self.roi(adj_close_benchmark[f"Adj Close_{self.benchmark_ticker}"])
        data["Profit"] = self.profit(self.total_amount, adj_close_benchmark[f"Adj Close_{self.benchmark_ticker}"])
        data["Daily Risk"] = self.risk(daily_return_benchmark[f"DailyReturn_{self.benchmark_ticker}"], False)
        data["Yearly Risk"] = self.risk(daily_return_benchmark[f"DailyReturn_{self.benchmark_ticker}"], True)
        data["Sharpe Ratio"] = round(self.sharpe_ratio(daily_return_benchmark[f"DailyReturn_{self.benchmark_ticker}"], 0), 3)
        df2 = pd.DataFrame([data]).transpose()
        df2 = df2.rename(columns = {0:"Benchmark"})

        df_merge = pd.concat([df,df2], axis = 1)

        if export_report:

            # create a new xlsx file
            filename = "PortfolioPerformanceReport.xlsx"
            writer = pd.ExcelWriter(filename, engine='xlsxwriter')

            # Export all data
            self.all_data().to_excel(writer, sheet_name='Time Series Data')

            # Export portfolio analysis data
            df_merge.to_excel(writer, sheet_name='Portfolio Overview')

            # Export Adj Close Prices - Normalization to 100 chart
            df_empty = pd.DataFrame()
            df_empty.to_excel(writer, sheet_name='Adjusted Close Prices')
            worksheet = writer.sheets['Adjusted Close Prices']
            self.normalized_graph()
            worksheet.insert_image('B2', 'ClosingPrices.png')

            # Export Daily Returns Histogram
            df_empty = pd.DataFrame()
            df_empty.to_excel(writer, sheet_name = 'Daily Returns Histogram')
            worksheet = writer.sheets['Daily Returns Histogram']
            self.graph_returns()
            worksheet.insert_image('B2', 'ReturnsHistogram.png')

            # Export Sharpe Ratio df and chart
            sharpe_df.to_excel(writer, sheet_name='Monte Carlo Simulation', startcol=15, startrow=2)
            worksheet = writer.sheets['Monte Carlo Simulation']
            worksheet.insert_image('B2', 'MC_Sharpe.png')

            writer.close()

        return df
