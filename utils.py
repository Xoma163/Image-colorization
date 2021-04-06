class CyclePercentWriter:
    """
    Логирует выполнение долгих методов
    """

    def __init__(self, max_size: int, per: int = 10):
        """

        :param max_size: размер списка по которому итетируемся
        :param per: через сколько процентов выводить информацию
        """
        self.per = per
        self.tick_every = 100 // self.per
        self.per_percents = [round(x / self.tick_every * max_size) for x in range(self.tick_every + 1)]
        self.per_percents[-1] -= 1  # 100% fix

    def check(self, iteration):
        try:
            index = self.per_percents.index(iteration)
            return index * self.per
        except ValueError:
            pass
        return None
