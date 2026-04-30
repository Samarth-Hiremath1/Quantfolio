class Event:
    """
    Base class providing an interface for all subsequent (inherited) events,
    that will trigger further events in the trading infrastructure.
    """
    pass

class MarketEvent(Event):
    """
    Handles the event of receiving a new market update with corresponding bars.
    """
    def __init__(self):
        self.type = 'MARKET'

class SignalEvent(Event):
    """
    Handles the event of sending a Signal from a Strategy object.
    This is received by a Portfolio object and acted upon.
    """
    def __init__(self, strategy_id: str, ticker: str, datetime, signal_type: str, strength: float):
        """
        :param strategy_id: Unique identifier for the strategy.
        :param ticker: The ticker symbol, e.g. 'AAPL'.
        :param datetime: Timestamp at which the signal was generated.
        :param signal_type: 'LONG' or 'SHORT'.
        :param strength: An adjustment factor "suggestion" to scale quantity, typically 1.0.
        """
        self.type = 'SIGNAL'
        self.strategy_id = strategy_id
        self.ticker = ticker
        self.datetime = datetime
        self.signal_type = signal_type
        self.strength = strength

class OrderEvent(Event):
    """
    Handles the event of sending an Order to an execution system.
    The order contains a ticker, quantity, direction, and type (mkt or lmt).
    """
    def __init__(self, ticker: str, order_type: str, quantity: int, direction: str):
        self.type = 'ORDER'
        self.ticker = ticker
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction

    def print_order(self):
        print(f"Order: {self.ticker} {self.direction} {self.quantity} {self.order_type}")

class FillEvent(Event):
    """
    Encapsulates the notion of a Filled Order, as returned
    from a brokerage. Stores the quantity of an instrument
    actually filled and at what price. In addition, stores
    the commission of the trade from the brokerage.
    """
    def __init__(self, timeindex, ticker: str, exchange: str, quantity: int, direction: str, fill_cost: float, commission: float = None):
        self.type = 'FILL'
        self.timeindex = timeindex
        self.ticker = ticker
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost
        
        # Calculate commission natively if none is natively provided
        if commission is None:
            self.commission = self.calculate_ib_commission()
        else:
            self.commission = commission

    def calculate_ib_commission(self):
        """
        Calculates the fees of trading based on an Interactive
        Brokers fee structure for API (e.g. $0.005 per share, minimum $1.00).
        """
        full_cost = 0.005 * self.quantity
        return max(full_cost, 1.00)
