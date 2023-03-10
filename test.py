import yfinance as yf

class Test:

    def __init__(self):
        pass

    data = yf.download("AAPL", start="2019-01-01", end="2023-02-28", interval = "1d")

    def returns(self):
        self.data["Epa"] = 1

        a = "KIKS"

        return self.data

    def t(self):
        return self.a

t = Test()
print(t.returns())

print(t.t())

