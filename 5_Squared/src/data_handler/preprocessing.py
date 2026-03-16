"""This is for the Z-scoring and Sigma Clipping of the data"""
import pandas as pd
import numpy as np

class DataTransformer:
    @staticmethod
    def sigma_clip(series: pd.Series, sigma=3) -> pd.Series:
        """
        Caps outliers that fall outside of the Mean +/- (sigma * STD), with a sigma of 3.
        """
        mean = series.mean()
        std = series.std()
        
        lower_bound = mean - (sigma * std)
        upper_bound = mean + (sigma * std)
        
        return series.clip(lower=lower_bound, upper=upper_bound)

    @staticmethod
    def z_score(series: pd.Series) -> pd.Series:
        """Standardizes data to mean 0 and std 1"""
        return (series - series.mean()) / series.std()

    @classmethod
    def scale(cls, series: pd.Series, sigma=3) -> pd.Series:
        """ 
        It sigma clips to remove outliers to then standardize with a Z-score
        """
        clipped_data = cls.sigma_clip(series, sigma=sigma)
        return cls.z_score(clipped_data)
    