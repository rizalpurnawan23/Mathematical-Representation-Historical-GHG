# Bismillahirrahmanirrahim
"""
Title               : 'greenhouse_emissions'
Dev & Author        : Rizal Purnawan
Date of creation    : 02/13/2024
Update              : 02/22/2024

Description
-----------

This script contains algorithms as the realization of the theory
presented in [1].
"""

# -------------------------------------------------------------------
# LIBRARIES IN USE:
from math import factorial
import numpy as np
import sympy as sp
import pandas as pd
from sklearn.linear_model import LinearRegression

# -------------------------------------------------------------------
# CLASS--MAIN ALGORITHMS:
class GreenhouseEmissionModel:
    """
    Description
    -----------
    The development of the ML models and the construction of the CEP
    is conducted with this class. The components of the CEP is
    developped sequentially. The properties of the CEP are also
    contained in this class.

    We leverage two powerful mathematical tools in computations,
    namely machine learning (ML) and computer algebra system (CAS).
    We will use 'LinearRegression' from 'sklearn.linear_model' for
    constructing the Bernstein polynomial of the CEP as an
    implementation of ML. On the other hand, we use library 'sympy',
    a CAS library, for algorithms of properies of the CEP in [1].
    """
    # INSTANCES:
    def __init__(self, df):
        """
        Description
        -----------
        The argument for 'df' shall be the cleaned dataset of the
        historical greenhouse gas emissions.
        """
        self.df = df
        self.countries = sorted(list(df["Country"].unique()))
        self.GHGs = list(df.columns[2:])
        self.years = list(df["Year"].unique())

    # r-NORM:
    def r_norm(self, X, Y):
        """
        Description
        -----------
        This instance method computes the r-norm [2] of two discrete
        random variables.
        """
        EX = np.array([np.mean(X)] *len(X))
        norm = np.linalg.norm
        return (
            norm(np.array(X) - np.array(Y)) /
            norm(np.array(X) - EX)
        )

    # GHG-COUNTRY FILTER:
    def ghg_country_filter(self, c, g):
        """
        Description
        -----------
        This instance method provides the slice of the dataset by the
        parameters 'c' (country) and 'g' (GHG type).
        """
        df = self.df.copy()
        fil_df = df[(df["Country"] ==  c)].loc[:, ["Year", g]].copy()
        fil_df = fil_df.reset_index(drop= True)
        return fil_df
    
    # DEP FUNCTION:
    def dep_function(self, c, g):
        """
        Description
        -----------
        This instance method computes the DEP function.

        Parameters:
        - c     : Country code.
        - g     : GHG type.
        """
        df = self.ghg_country_filter(c, g)
        return df.set_index("Year").squeeze()

    # CONDITIONAL EXPECTATION:
    def conditional_expectation(self, X, S):
        """
        Description
        -----------
        This instance method computes the conditional expectation of
        a discrete random variable 'X' on an event 'S'. 

        Parameters:
        - X     : A pandas series representing a random variable.
        - S     : A list whose values are contained in the index of
                  X.
        """
        X_on_S = X.loc[S]
        return X_on_S.sum() / len(S)
    
    # BINOMIAL COEFFICIENT:
    def binomial(self, n, k):
        """
        Description
        -----------
        This instance method computes binomal coefficients, which
        will be necessary for constructing the Bernstein polynomial
        for the continuous emission process (CEP).

        Parameters:
        - n     : A nonnegative integer.
        - k     : A nonnegative integer.
        """
        return factorial(n) / (factorial(k) *factorial(n - k))

    # MOVING AVERAGE:
    def ma(self, w, X):
        """
        Description
        -----------
        This instance method computes the map 'MA' in postulate 2.

        Parameters:
        - w     : The radius, a float or integer.
        - X     : A pandas series representing a random variable.
        """
        domain = X.index
        MAwX = list()
        for x in domain:
            Bw = [u for u in domain if x - w < u < x + w]
            MAwX.append(self.conditional_expectation(X, Bw))
        return pd.Series(MAwX, index= domain)
    
    # COMPUTING INFIMUM AND SUPREMUM OF FUNCTION:
    def func_inf_sup(self, func, interval, step= 0.01):
        """
        Description
        -----------
        This instance method computes the numerical minimum and
        maximum of a function on given domain.

        Parameters:
        - func      : A Python function.
        - interval  : A SymPy interval.
        - step      : A small float representing the increment for
                      numerical computations.
        """
        inf_int, sup_int = interval.inf, interval.sup
        numeric_interval = np.arange(
            float(inf_int), float(sup_int) + step, step= step
            )
        func_interval = [func(t) for t in numeric_interval]
        return {
            "inf": min(func_interval), "sup": max(func_interval)
        }

    # CONTINUOUS EMISSION PROCESS:
    def cep_function(self, c, g, w= 5, n= 100):
        """
        Description
        -----------
        This instance method computes the continuous emission process
        (CEP) given a country 'c' and a GHG 'g'. The output is a
        Python function with the time domain T.

        Parameters:
        - c     : A string representing the country code.
        - g     : A string representing the GHG code.
        - n     : A positive integer representing the degree of
                  Bernstein polynomial.
        """
        df = self.ghg_country_filter(c, g).copy()
        df.reset_index(drop= True, inplace= True)
        y = self.ma(w, df[g])
        X = list()
        for k in range(n + 1):
            X.append(
                [
                    self.binomial(n, k)
                    *(0.001 *t)**k *(1 -0.001 *t)**(n - k)
                    for t in df["Year"]
                ]
            )
        X = np.transpose(np.array(X))
        model = LinearRegression(fit_intercept= False)
        _ = model.fit(X, y)
        coeffs = model.coef_
        pre_CEP = lambda t: sum(
            [
                b *self.binomial(n, k)
                *(0.001 *t)**k *(1 -0.001 *t)**(n - k)
                for b, k in zip(coeffs, range(n + 1))
            ]
        )
        inf_CEP = self.func_inf_sup(
            pre_CEP,
            sp.Interval(min(self.years), max(self.years))
            )["inf"]
        if inf_CEP < 0:
            CEP = lambda t: pre_CEP(t) - inf_CEP
        else:
            CEP = pre_CEP
        return CEP
    
    # FINDING CRITICAL POINTS:
    def critical_points(self, func, domain, delta):
        """
        Description
        -----------
        This instance method computes the critical points given a
        function 'func' with the domain or a restriction. The
        computation is conducted numerically.

        Parameters:
        - func      : A Python function.
        - domain    : The domain or a restriction for the function.
        - delta     : A float or integer for constructing open balls
                      on the domain.
        """
        crit_pts = list()
        for p in domain:
            B_delta_p = [
                x for x in domain if p -delta < x < p +delta
                ]
            if all(func(x) < func(p) for x in B_delta_p if x != p) \
                    and all(
                        func(x) > func(p)
                        for x in B_delta_p if x != p
                        ):
                crit_pts.append(p)
        return crit_pts

    # HISTORICAL UPPER BOUND EMISSION (HUBE):
    def hube(self, CEP, rho= 5, step= 0.1):
        """
        Description
        -----------
        This instance method computes the Historical Upper Bound
        Emission (HUBE).

        Parameters:
        - CEP           : A Python function representing the CEP.
        - rho           : A float representing a radius of an open
                          ball [1], or a 'None'. If 'None' is used,
                          then the time of HUBE is returned.
        - step          : A small float representing the increment
                          for numerical computation.
        """
        year_min, year_max =min(self.years), max(self.years)
        numeric_T = np.arange(year_min, year_max + step, step= step)
        CEP_list = [CEP(t) for t in numeric_T]
        t_ub = [
            t for t, c in zip(numeric_T, CEP_list)
            if c == max(CEP_list)
            ][0]
        
        if rho is None:
            return t_ub
        else:
            T = sp.Interval(year_min, year_max)
            return sp.Interval.open(
                t_ub -rho, t_ub +rho).intersect(T)

    # HISTORICAL PEAK EMISSION (HPE):
    def hpe(self, CEP, rho= 5, delta= 0.1):
        """
        Description
        -----------
        This instance method computes the Historical Peak Emission
        (HPE).

        Parameters:
        - CEP   : A Python function representing the CEP.
        - rho   : A float representing the radius of an open ball.
        """
        a, b = min(self.years), max(self.years)
        T = sp.Interval(a, b)

        numeric_T = list(np.arange(a, b, step= delta)) + [b]
        CEP_num_T = [CEP(t) for t in numeric_T]
        tp = [
            t for t, c in zip(numeric_T, CEP_num_T)
            if c == max(CEP_num_T)
            ][0]
        if tp in [a, b]:
            return None
        else:
            return sp.Interval.open(tp - rho, tp + rho).intersect(T)

        
    # PERIOD OF RAPID GROWING EMISSION:
    def rge(self, CEP, rho= 5, step= 0.1):
        """
        Description
        -----------
        This instance method computes both the time and period of
        rapid growing emission (RGE).

        Parameters:
        - CEP   : A Python function representing the CEP.
        - rho   : A positive float representing the radius of open
                  ball of the period of RGE.
        - step  : Numerical increment for numerical computation.
        """
        T = sp.Interval(min(self.years), max(self.years))
        num_int_T = np.arange(
            min(self.years) + step,
            max(self.years),
            step= step
            )
        t = sp.symbols("t")
        dCEP = sp.lambdify(t, sp.diff(CEP(t), t), "numpy")
        t_ub = float(self.hube(CEP, rho= None))
        y = t_ub -step
        tg = None
        while y > min(num_int_T):
            B_rho_y = [x for x in num_int_T if y -rho < x < y +rho]
            if all(dCEP(x) < dCEP(y) for x in B_rho_y if x != y):
                tg = y
                break
            y = y -step
        if tg is not None:
            period_RGE = sp.Interval.open(
                tg -rho, tg +rho).intersect(T)
            return {
                "time-RGE": tg,
                f"{float(2 *rho)}-period-RGE": period_RGE
            }
        else:
            return {
                "time-RGE": None,
                f"{float(2 *rho)}-period-RGE": None
            }
        
    # PERIOD OF RAPID SHRINKING EMISSION:
    def rse(self, CEP, rho= 5, step= 0.1):
        """
        Description
        -----------
        This instance method computes both the time and period of
        rapid shrinking emission (RSE).

        Parameters:
        - CEP   : A Python function representing the CEP.
        - rho   : A positif float representing the radius of open
                  ball of the period of RGE.
        - step  : Numerical increment for numerical computation.
        """
        T = sp.Interval(min(self.years), max(self.years))
        num_int_T = np.arange(
            min(self.years) + step,
            max(self.years),
            step= step
            )
        t = sp.symbols("t")
        dCEP = sp.lambdify(t, sp.diff(CEP(t), t), "numpy")
        t_ub = float(self.hube(CEP, rho= None))
        y = t_ub +step
        ts = None
        while y < max(num_int_T):
            B_rho_y = [x for x in num_int_T if y -rho < x < y +rho]
            if all(dCEP(x) > dCEP(y) for x in B_rho_y if x != y):
                ts = y
                break
            y = y + step
        if ts is not None:
            period_RGE = sp.Interval.open(
                ts -rho, ts +rho).intersect(T)
            return {
                "time-RSE": ts,
                f"{float(2 *rho)}-period-RSE": period_RGE
            }
        else:
            return {
                "time-RSE": None,
                f"{float(2 *rho)}-period-RSE": None
            }

    # NUMERIC INPIVIOD:
    def piviods(self, CEP, rho= 5, step= 0.001, K= 1):
        """
        Description
        -----------
        This instance computes the inpiviods of a CEP. We employ
        numerical method instead of symbolic computation for
        determining the inpiviods due to the computational expense
        of the latter.

        Parameters:
        - CEP       : A Python function representing the CEP.
        - rho       : A float or integer representing the radius of
                    the inpiviod open ball.
        - step      : A small float for iteration in finding
                    solutions.
        """
        min_year, max_year = min(self.years), max(self.years)
        T = sp.Interval(min_year, max_year)
        
        nu_id = lambda t: (t - min_year) / (max_year - min_year)
        inv_nu_id = lambda x: x *(max_year - min_year) + min_year
        
        inf_sup_CEP = self.func_inf_sup(CEP, T)
        inf_CEP, sup_CEP = inf_sup_CEP.values()
        nu_CEP = lambda t: (CEP(t) - inf_CEP) / (sup_CEP - inf_CEP)

        Phi = lambda x: nu_CEP(inv_nu_id(x))

        t_ub = self.hube(CEP, rho= None)
        tg = self.rge(CEP)["time-RGE"]
        ts = self.rse(CEP)["time-RSE"]
        ui = t_ub if tg is None else tg
        ud = None if ts is None else ts

        df = pd.DataFrame(
            {
                "unit": np.arange(0, 1 + step, step= step),
                "Phi": [
                    Phi(x)
                    for x in np.arange(0, 1 + step, step= step)
                ]
            }
        )

        # Testing short map:
        is_short_map = [np.nan]
        for k in df.index[1:]:
            if (
                    abs(df["Phi"][k] - df["Phi"][k -1]) /
                    abs(df["unit"][k] - df["unit"][k -1])
                    ) <= K:
                is_short_map.append(True)
            else:
                is_short_map.append(False)
        df["is_short_map"] = is_short_map
        df = df.dropna()

        # Computing inpiviod:
        Si = df[
            (df["is_short_map"] == True) &
            (df["unit"] < nu_id(ui))
            ]["unit"]
        if len(Si) == 0:
            inpiviod = None
        else:
            vi = inv_nu_id(Si.max())
            inpiviod = sp.Interval.open(vi -rho, vi +rho).intersect(T)

        # Computing depiviod:
        if ud is None:
            depiviod = None
        else:
            Sd = df[
                (df["is_short_map"] == True) &
                (df["unit"] > nu_id(ud))
                ]["unit"]
            if len(Sd) == 0:
                depiviod = None
            else:
                vd = inv_nu_id(Sd.min())
                depiviod = sp.Interval.open(
                    vd -rho, vd +rho).intersect(T)

        # The output:
        return {
            f"{float(2 *rho)}-inpiviod": inpiviod,
            f"{float(2 *rho)}-depiviod": depiviod
        }
    
    # HISTORICAL EXPECTED GROWTH RATE (HEGR):
    def hegr(self, c, g):
        """
        Description
        -----------
        This instance method computes the historical expected growth
        rate (HEGR) of the DEP.

        Parameters:
        - c     : A string representing the country code.
        - g     : A string representing the GHG type.
        """
        DEP = list(self.dep_function(c, g))
        return (DEP[-1] - DEP[0]) / (len(DEP) -1)
        
    
    # CONDITIONAL HISTORICAL EXPECTED GROWTH RATE (HEGR):
    def conditional_hegr(self, c, g, I):
        """
        Description
        -----------
        This instance method computes the conditional HEGR given a
        discrete subinterval I.
        
        Parameters:
        - c     : A string representing the country code.
        - g     : A string representing the GHG type.
        - I     : A numeric discrete interval within the time domain.
        """
        DEP_I = list(self.dep_function(c, g).loc[I])
        return (DEP_I[-1] - DEP_I[0]) / (len(DEP_I) -1)
    
    # THE b-TRANSFORMER:
    def b_transformer(self, CEP, DEP, A):
        """
        Description
        -----------
        This instance method computes the map 'b' in definition 12
        in [1].

        Parameters:
        - CEP   : A Python function representing the CEP.
        - DEP   : A pandas series representing the DEP.
        - A     : A list or numpy array representing the conditioning
                  measurable set for CLBT and CUBT.
        """
        Xcg = pd.Series(
            [CEP(t) for t in DEP.index], index= DEP.index
        )
        DEP_min_CEP = DEP - Xcg
        abs_DEP_min_CEP = DEP_min_CEP.abs()
        return abs_DEP_min_CEP.loc[A].max()

    # CONDITIONAL LOWER BOUND TRANSFORMATION (CLBT):
    def clbt(self, CEP, DEP, A):
        """
        Description
        -----------
        Thin instance method computes the conditional lower bound
        trainsformation (CLBT) in accordance with definition 12 in
        [1].

        Parameters:
        - CEP   : A Python function representing the CEP.
        - DEP   : A pandas series representing the DEP.
        - A     : A list or numpy array representing the 
        """
        return lambda t: CEP(t) - self.b_transformer(CEP, DEP, A)
    
    # CONDITIONAL UPPER BOUND TRANSFORMATION (CUBT):
    def cubt(self, CEP, DEP, A):
        """
        Description
        -----------
        Thin instance method computes the conditional upper bound
        trainsformation (CUBT) in accordance with definition 12 in
        [1].

        Parameters:
        - CEP   : A Python function representing the CEP.
        - DEP   : A pandas series representing the DEP.
        - A     : A list or numpy array representing the 
        """
        return lambda t: CEP(t) + self.b_transformer(CEP, DEP, A)
    
    # HISTORICAL LOWER BOUND TRANSFORMATION (HLBT):
    def hlbt(self, CEP, DEP):
        """
        Description
        -----------
        This instance method computes the historical lower bound
        transformation (hlbt) in accordance with definition 12 in
        [1].

        Parameters:
        - CEP   : A Python function representing the CEP.
        - DEP   : A pandas series representing the DEP.
        """
        years = self.years
        return lambda t: CEP(t) - self.b_transformer(CEP, DEP, years)
    
    # HISTORICAL UPPER BOUND TRANSFORMATION (HLBT):
    def hubt(self, CEP, DEP):
        """
        Description
        -----------
        This instance method computes the historical upper bound
        transformation (hlbt) in accordance with definition 12 in
        [1].

        Parameters:
        - CEP   : A Python function representing the CEP.
        - DEP   : A pandas series representing the DEP.
        """
        years = self.years
        return lambda t: CEP(t) + self.b_transformer(CEP, DEP, years)

    
"""
REFERENCE
[1] Purnawan, R. (2024). "A Mathematical Representation of Historical
    Greenhouse Gas Emissions with Rigorous Insights". TBD

[2] Purnawan, R. (2023). "An Exploration on a Normed Space Called
    r-Normed Space: Some Properties and an Application".
    MDPI preprints.org
"""