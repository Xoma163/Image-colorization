import logging
import time
from datetime import timedelta, datetime

from consts import LOGS_INFO_FILE


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


def get_time_str(_time):
    """
    Выводит время выполнения фразой
    :param _time: секунды
    :return:
    """
    sec = timedelta(seconds=int(_time))
    d = datetime(1, 1, 1) + sec

    day_str = f"{d.day - 1} {decl_of_num(d.day - 1, ['день', 'дня', 'дней'])} " if (d.day - 1) else ""
    hour_str = f"{d.hour} {decl_of_num(d.hour, ['час', 'часа', 'часов'])} " if d.hour else ""
    minute_str = f"{d.minute} {decl_of_num(d.minute, ['минута', 'минуты', 'минут'])} " if d.minute else ""
    second_str = f"{d.second} {decl_of_num(d.second, ['секунда', 'секунды', 'секунд'])} " if d.second else ""
    time_str = f"{day_str}{hour_str}{minute_str}{second_str}".strip()
    return time_str


def lead_time_writer(function):
    """
    Декоратор который выводит время выполнения метода
    :param function:
    :return:
    """

    def wrapper(*args, **kwargs):
        time_start = time.time()
        func = function(*args, **kwargs)
        time_end = time.time()

        logger.info(f"Время выполнения - {get_time_str(time_end - time_start)}")
        return func

    return wrapper


def decl_of_num(number, titles):
    """
    Склоняет существительное после числительного
    number: число
    titles: 3 склонения
    """
    cases = [2, 0, 1, 1, 1, 2]
    if 4 < number % 100 < 20:
        return titles[2]
    elif number % 10 < 5:
        return titles[cases[number % 10]]
    else:
        return titles[cases[5]]


_log_format = "%(asctime)s - [%(levelname)-8s] - %(name)-15s:%(lineno)-3d - %(message)s"


# _log_format = "%(asctime)s - [%(levelname)s] - %(name)s - %(filename)s:%(lineno)d - %(message)s"


def get_logger(name):
    def get_stream_handler():
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(logging.Formatter(_log_format))
        return stream_handler

    def get_file_handler():
        file_handler = logging.FileHandler(LOGS_INFO_FILE, encoding="UTF-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_log_format))
        return file_handler

    _logger = logging.getLogger(name)
    _logger.setLevel(logging.DEBUG)
    _logger.addHandler(get_file_handler())
    _logger.addHandler(get_stream_handler())
    return _logger


logger = get_logger(__name__)
