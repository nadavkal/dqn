class Action(object):
    def __init__(self):
        self.actions_history = []
        self.last_action = None

    def update_last_action(self):
        self.last_action = self.actions_history[-1]

    def buy(self, price, cash, holding):
        amount = int(cash/price)
        result = {"action": "buy", "amount": amount, "price": price,
                  "cash": cash - (int(amount * price)), "holding": holding + amount}
        self.actions_history.append(result)
        self.update_last_action()

    def sell(self, price, cash, holding):
        cash = cash + (price * holding)
        result = {"action": "sell", "amount": holding, "price": price,
                  "cash": cash, "holding": 0}
        self.actions_history.append(result)
        self.update_last_action()

    def hold(self, price, cash, holding):
        result = {"action": "hold", "amount": 0, "price": price,
                  "cash": cash, "holding": holding}
        self.actions_history.append(result)
        self.update_last_action()