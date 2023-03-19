from classes import Stock, Portfolio

import warnings
warnings.filterwarnings("ignore")

start = "2017-02-27"
end = "2023-03-18"

stock1 = Stock("MSFT", "Microsoft", 1000, start, end)
stock2 = Stock("AAPL", "Apple", 1000, start, end)
stock3 = Stock("BA", "Boeing", 1000, start, end)
stock4 = Stock("COST", "Costco", 1000, start, end)
stock5 = Stock("CS", "Credit Suisse", 1000, start, end)

portfolio = Portfolio([stock1, stock2, stock3, stock4, stock5], start, end, "^GSPC")

portfolio.performance_report()

