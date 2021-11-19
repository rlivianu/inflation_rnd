import numpy as np
import pandas as pd
from datetime import date


def split_strikes(strikes: np.ndarray) -> (np.ndarray, np.ndarray):
    """Splits array of strikes into cap and floor strikes"""
    index = 0
    for i in range(len(strikes) - 1):
        if strikes[i] > strikes[i+1]:
            index = i + 1
    return strikes[:index], strikes[index:]


def bond_price_from_ois(ois_rate: float,
                        maturity: float) -> float:
    """Returns implied bond price from OIS rate for given maturity"""
    return 1 / (maturity * ois_rate / 100 + 1)


def cap_parity(floor_prices: np.ndarray,
               floor_strikes: np.ndarray,
               swap_rate: float,
               bond_price: float,
               maturity: float) -> np.ndarray:
    """Given floor prices, returns the cap prices induced by put-call parity"""
    cap_prices = (floor_prices / 10000 + (bond_price * (1 + swap_rate/100)**maturity)
                  - ((1 + floor_strikes)**maturity) * bond_price) * 10000
    return cap_prices


def option_indices(swap_rate: float,
                   floor_strikes: np.ndarray,
                   cap_strikes: np.ndarray) -> (list, list):
    """Return the indices of out-the-money options"""
    floor_indices = [i for i in range(sum(floor_strikes < swap_rate))]
    cap_indices = [i for i in range(len(cap_strikes) - sum(cap_strikes > swap_rate), len(cap_strikes))]
    return floor_indices, cap_indices
