**Investment Analytics**


This repository is creates an Excel report analyzing a portfolio comprised by a given set of stocks.
The code resorts to the Yfinance API to retrieve the data. Hence the tickers should match the tickers form Yahoo Finance.

To generate a report perform the following steps:

1) Select a start and end date matching "yyyy-mm-dd" format":

`start = "2017-02-27" `

`end = "2023-03-18"`

2) Define instances of the Stock object containing information about the ticker, name, start and end date:

`stock1 = Stock("MSFT", "Microsoft", 1000, start, end)`

`stock2 = Stock("AAPL", "Apple", 1000, start, end)`

`stock3 = Stock("BA", "Boeing", 1000, start, end)`

`stock4 = Stock("COST", "Costco", 1000, start, end)`

`stock5 = Stock("CS", "Credit Suisse", 1000, start, end)`

3) Define an instance of the Portfolio object by passing a list of stock instance objects, a date range and a Benchmark ticker:

`portfolio = Portfolio([stock1, stock2, stock3, stock4, stock5], start, end, "^GSPC")`

4) Finally call the performance_report method. It will generate an Excel file _PortfolioPerformanceReportx.xlsx_

`portfolio.performance_report()`

**Note**: In the generated performance report, metrics with no suffix refer to the portfolio. In above example, Adj_Close refers to the adjusted close price of the portfolio. Conversely, Adj_Close_MSFT refers to the adjusted closed price of Microsoft