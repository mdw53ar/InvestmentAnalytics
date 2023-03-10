from classes import Investor, Stock, Portfolio

import warnings
warnings.filterwarnings("ignore")

investor = Investor("Hugo", "De Elejabeitia Agudo")
stock1 = Stock("MSFT", "Microsoft", 30000, "2013-02-27", "2023-02-24")
stock2 = Stock("AAPL", "Apple", 2000, "2013-02-27", "2023-02-24")
stock3 = Stock("BA", "Boeing", 20000, "2013-02-27", "2023-02-24")
stock4 = Stock("COST", "Costco", 20000, "2013-02-27", "2023-02-24")
stock5 = Stock("CS", "Costco", 20000, "2013-02-27", "2023-02-24")

portfolio = Portfolio([stock1, stock2, stock3, stock4, stock5], start ="2017-02-27",
                      end = "2023-02-24", benchmark_ticker = "^GSPC")

daily_return = portfolio.all_data()["DailyReturn"]
adj_close = portfolio.all_data()["Adj Close"]
total_amount = portfolio.total_amount

print(portfolio.performance_report(portfolio.total_amount,daily_return,adj_close ))
