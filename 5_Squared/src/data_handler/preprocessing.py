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
    

    #SEE IF THIS WORKS

    """@staticmethod
    def signed_l3_composite(df: pd.DataFrame) -> pd.Series:
        
        Calculates a directional L3 score.
        Positive values are amplified, negative values are penalized.
        
        # Cube the values (preserves sign)
        # x^3 makes 0.8 -> 0.512 and -0.8 -> -0.512
        cubed_sum = np.sum(np.power(df, 3), axis=1)
        
        # Take the cube root while preserving the sign of the sum
        # We use (abs(x)^(1/3)) * sign(x) to avoid math errors with negative roots
        final_scores = np.sign(cubed_sum) * np.power(np.abs(cubed_sum), 1/3)
        
        return pd.Series(final_scores, index=df.index, name="directional_l3")"""