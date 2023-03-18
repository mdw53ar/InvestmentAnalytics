from classes import Investor, Stock, Portfolio

import warnings
warnings.filterwarnings("ignore")

investor = Investor("Hugo", "De Elejabeitia Agudo")
stock1 = Stock("MSFT", "Microsoft", 1000, "2017-02-27", "2023-03-18")
stock2 = Stock("AAPL", "Apple", 1000, "2017-02-27", "2023-03-18")
stock3 = Stock("BA", "Boeing", 1000, "2017-02-27", "2023-03-18")
stock4 = Stock("COST", "Costco", 1000, "2017-02-27", "2023-03-18")
stock5 = Stock("CS", "Credit Suisse", 1000, "2017-02-27", "2023-03-18")

portfolio = Portfolio([stock1, stock2, stock3, stock4, stock5], start ="2017-02-27",
                      end = "2023-03-18", benchmark_ticker = "^GSPC")

print(portfolio.performance_report())

