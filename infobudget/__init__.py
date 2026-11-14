"""infobudget: information lower bound of health indicators for remaining useful life prediction.

Reference implementation of Le and Nguyen, "Information Lower Bounds of Trajectory Health Indicators for Remaining Useful Life Prediction".
"""
from .bound import LifetimeModel, information_lower_bound, rmse_ratio_from_gain, rmse_floor, bootstrap_units
from .indicators import (fit_ridge_hi, fit_pca_hi, trajectory_features, sensor_features, geometric_scores, monotonicity, trendability,
                         prognosability, template_shrinkage, shrinkage_constructor, causal_mean, causal_slope)
from .pipeline import Config, BoundReport, evaluate, fleet_size_curve, cv_pairs
from .synthetic import wiener_fleet
from .io import load_csv

__version__ = "1.0.0"
__all__ = [n for n in dir() if not n.startswith("_")]
