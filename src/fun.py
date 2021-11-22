import numpy as np
import pandas as pd
from datetime import date
from scipy.stats import norm
from scipy import optimize


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


def points_convex(x: np.ndarray,
                  y: np.ndarray) -> bool:
    """Determines if the (x, y) coordinates given lie on the graph of a convex function"""
    for i in range(1, len(x) - 1):
        if signed_triangle_area(x[i-1:i+2], y[i-1:i+2]) > 0:
            return False
    return True


def signed_triangle_area(x: np.ndarray,
                         y: np.ndarray) -> float:
    """Return the signed area of a triangle determined by vertices x and y (both length 3)"""
    return x[0] * (y[2] - y[1]) + x[1] * (y[0] - y[2]) + x[2] * (y[1] - y[0])


def clean_caps(cap_strikes: np.ndarray,
               cap_prices: np.ndarray) -> (np.ndarray, np.ndarray):
    """Removes missing and non-convex data from cap prices"""
    new_strikes = cap_strikes[cap_prices == cap_prices]
    new_caps = cap_prices[cap_prices == cap_prices]

    if points_convex(new_strikes, new_caps):
        return new_strikes, new_caps
    else:
        indices_to_remove = []
        for i in range(1, len(new_strikes) - 1):
            if signed_triangle_area(new_strikes[i - 1:i + 2], new_caps[i - 1:i + 2]) > 0:
                indices_to_remove.append(i)
        return np.delete(new_strikes, indices_to_remove), np.delete(new_caps, indices_to_remove)


def bs_call(spot, strike, expiry, r, sigma):
    """ Computes the true value of a European call option under Black-Scholes assumptions
    :param spot: float
        The spot price of the asset
    :param strike: float
        The strike price of the option
    :param expiry: float
        The time to maturity of the option
    :param r: float
        The risk-free rate
    :param sigma: float
        The volatility of the asset
    :return: float
        The value of the option
    """
    d1 = (np.log(spot / strike) + (r + sigma ** 2 / 2) * expiry) / (sigma * np.sqrt(expiry))
    d2 = d1 - sigma * np.sqrt(expiry)
    return spot * norm.cdf(d1) - strike * np.exp(-r * expiry) * norm.cdf(d2)


def implied_volatility(cap_strikes, cap_prices, bond_price, swap_rate, maturity):
    """Computes implied Black-Scholes volatility from inflation caps and floors"""

    def objective(sigma, value, spot, strike, expiry, r):
        return bs_call(spot, strike, expiry, r, sigma) - value

    values = cap_prices / (10000 * bond_price)
    spot = (1 + swap_rate) ** maturity
    strikes = (1 + cap_strikes) ** maturity
    x0 = np.ones(len(values)) * 0.1
    mats = np.ones(len(values)) * maturity

    out = optimize.newton(objective, x0=x0, args=(values, spot, strikes, 10, 0))
    return out