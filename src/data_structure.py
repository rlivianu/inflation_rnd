from .helpers import *
import pandas as pd
from datetime import date


class InflationData:
    """Loads ands stores inflation market data needed for RND estimation"""

    def __init__(self, option_fnames: list,
                 swil_fname: str,
                 ois_fname: str,
                 int_dates: bool = True,
                 start_date: int = 693594):
        # read in first sheet to get struct info
        test_sheet = pd.read_csv(option_fnames[0], header=None)

        # strikes
        self.strikes = test_sheet.iloc[0, :][1:].to_numpy()
        self.cap_strikes, self.floor_strikes = split_strikes(self.strikes)
        self.num_caps, self.num_floors = len(self.cap_strikes), len(self.floor_strikes)

        # num dates
        self.num_dates = test_sheet.dropna(subset=[0]).shape[0] - 1

        self.num_years = len(option_fnames)

        # read in all sheets
        self.options, self.caps, self.floors = [], [], []
        for fname in option_fnames:
            sheet = pd.read_csv(fname, na_values=['#VALUE!', '#DIV/0!']).iloc[:self.num_dates]
            if int_dates:
                sheet.iloc[:, 0] = sheet.iloc[:, 0].astype(int).apply(lambda x: date.fromordinal(x + start_date))
            sheet.columns = ['Date'] + [i for i in range(1, sheet.shape[1])]
            self.options.append(sheet)
            cap_sheet = sheet.iloc[:, :self.num_caps + 1]
            cap_sheet.columns = ['Date'] + self.cap_strikes.tolist()

            floor_sheet = sheet.iloc[:, :1].join(sheet.iloc[:, 1 + self.num_caps:])
            floor_sheet.columns = ['Date'] + self.floor_strikes.tolist()
            self.caps.append(cap_sheet)
            self.floors.append(floor_sheet)

        self.swil = pd.read_csv(swil_fname).iloc[:self.num_dates, 1:]
        self.ois = pd.read_csv(ois_fname).iloc[:self.num_dates, 1:11]
        self.bonds = self.ois
        for i in range(self.ois.shape[1]):
            self.bonds.iloc[:, i] = self.ois.iloc[:, i].apply(lambda x: bond_price_from_ois(x, i + 1))

        self.combined_caps = []
        self.implied_vols = []
        self.timplied_vols = []

    def compute_implied_vol(self):
        self.combined_caps = []
        self.implied_vols = []
        self.timplied_vols = []

        # Compute the implied caps from floors
        implied_caps = []
        for i, floor_df in enumerate(self.floors):
            aux_df = floor_df.copy()
            aux_df.insert(1, 'bond_price', self.bonds.iloc[:, i])
            aux_df.insert(2, 'swap_rate', self.swil.iloc[:, i])
            tf = lambda x: cap_parity(x.iloc[3:], self.floor_strikes, x.swap_rate, x.bond_price, i + 1)
            implied_caps.append(aux_df.apply(tf, 1))
        print('Floors converted into caps')

        # Combine implied caps with actual caps

        for i, implied_cap in enumerate(implied_caps):
            aux_df = pd.concat([implied_caps[i], self.caps[i].iloc[:, 1:]], axis=1)
            aux_df.insert(0, 'bond_price', self.bonds.iloc[:, i])
            aux_df.insert(1, 'swap_rate', self.swil.iloc[:, i])
            tf = lambda x: x.iloc[df_option_indices(x.swap_rate / 100, self.floor_strikes, self.cap_strikes)]
            self.combined_caps.append(aux_df.apply(tf, 1))
        print('Out-of-the-money options chosen')

        # Implied volatility
        for i, combined_cap in enumerate(self.combined_caps):
            print(i)
            aux_df = combined_cap.copy()
            aux_df.insert(0, 'bond_price', self.bonds.iloc[:, i])
            aux_df.insert(1, 'swap_rate', self.swil.iloc[:, i])
            tf = lambda x: pd.Series(implied_volatility(x.iloc[2:].index.to_numpy().astype(float),
                                                        x.iloc[2:].to_numpy().astype(float), x.bond_price,
                                                        x.swap_rate / 100, i + 1))
            out = aux_df.apply(tf, 1)
            out.columns = self.combined_caps[i].columns
            self.implied_vols.append(out)
            tivs = out.apply(lambda x: np.exp(x ** 2))
            tivs.columns = self.combined_caps[i].columns
            self.timplied_vols.append(tivs)
        print('Implied volatility computed')

        return None

    def fit_splines(self):
        self.splines = []
        for i in range(len(self.combined_caps)):
            print(i)
            aux_df = pd.concat([self.combined_caps[i], self.timplied_vols[i]], axis=1)
            n = self.combined_caps[i].shape[1]
            tf = lambda x: clean_fit_spline((1 + x.index.to_numpy()[:n]) ** (i + 1), x.iloc[:n].to_numpy(),
                                            x.iloc[n:].to_numpy())
            self.splines.append(aux_df.apply(tf, 1))
        return None

    def rnd(self,
            x: np.ndarray,
            day: int,
            year: int) -> np.ndarray:
        """Computes the value of the RND at given strikes for a date/maturity combination"""
        assert self.splines

        if np.isscalar(x):
            x = np.array([x])
        spline = self.splines[year - 1].iloc[day]
        y = eval_spline(x, spline)
        dy = eval_spline(x, spline, 1)
        ddy = eval_spline(x, spline, 2)
        sr = self.swil.iloc[day, year - 1] / 100
        return rnd_from_tiv(x, y, dy, ddy, year, sr)