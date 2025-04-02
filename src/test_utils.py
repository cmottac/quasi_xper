import numpy as np
import pandas as pd

def generate_trend_dataset(start_date_pre, end_date_pre, start_date_post, 
                           end_date_post, treatment_units, control_units,
                           metric_slope, treatment_shift, random_noise_scale, shifts):
    """
    Generate a synthetic dataset for unit testing with multiple treatment units.
    
    Parameters:
    - start_date_pre: str, start date of the pre-treatment period.
    - end_date_pre: str, end date of the pre-treatment period.
    - start_date_post: str, start date of the post-treatment period.
    - end_date_post: str, end date of the post-treatment period.
    - treatment_units: list of str, names of the treatment units.
    - control_units: list of str, names of the control units.
    - metric_slope: float, linear trend slope.
    - treatment_shift: float, shift applied post-treatment.
    - random_noise_scale: float, scale of the random noise.
    - shifts: dict, shift values per unit.
    
    Returns:
    - pd.DataFrame, synthetic dataset.
    """
    pre_period = pd.date_range(start=start_date_pre, end=end_date_pre)
    post_period = pd.date_range(start=start_date_post, end=end_date_post)
    
    def generate_unit_data(unit_name, is_treated=False):
        pre_metrics = metric_slope * np.arange(len(pre_period))
        post_metrics = metric_slope * np.arange(len(pre_period), len(pre_period) + len(post_period))
        if is_treated:
            post_metrics += treatment_shift
        metrics = np.concatenate([pre_metrics, post_metrics])
        timestamps = np.concatenate([pre_period, post_period])
        return pd.DataFrame({"timestamp": timestamps, "unit": unit_name, "metric": metrics})
    
    treated_data = pd.concat([generate_unit_data(unit, is_treated=True) for unit in treatment_units])
    control_data = pd.concat([generate_unit_data(unit) for unit in control_units])
    df = pd.concat([treated_data, control_data]).reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).sort_values(by='timestamp').reset_index(drop=True)
    
    for unit in df.unit.unique():
        shift = shifts.get(unit, 0)
        n_rows = df[df.unit == unit].shape[0]
        df.loc[df.unit == unit, 'metric'] += shift + np.random.normal(0, random_noise_scale, n_rows)
    
    return df

def generate_dataset_sine(start_date_pre, end_date_pre, start_date_post, end_date_post, treatment_units,
                          control_units, treatment_shift, random_noise_scale, shifts):
    """
    Generate a synthetic dataset using a sine wave for the metric values with multiple treatment units.
    
    Parameters:
    - start_date_pre: str, start date of the pre-treatment period.
    - end_date_pre: str, end date of the pre-treatment period.
    - start_date_post: str, start date of the post-treatment period.
    - end_date_post: str, end date of the post-treatment period.
    - treatment_units: list of str, names of the treatment units.
    - control_units: list of str, names of the control units.
    - treatment_shift: float, shift applied post-treatment.
    - random_noise_scale: float, scale of the random noise.
    - shifts: dict, shift values per unit.
    
    Returns:
    - pd.DataFrame, synthetic dataset.
    """
    pre_period = pd.date_range(start=start_date_pre, end=end_date_pre)
    post_period = pd.date_range(start=start_date_post, end=end_date_post)
    
    def generate_unit_data(unit_name, is_treated=False):
        pre_metrics = np.sin(np.linspace(0, 2 * np.pi * len(pre_period) / 7, len(pre_period)))
        post_metrics = np.sin(np.linspace(0, 2 * np.pi * len(post_period) / 7, len(post_period)))
        if is_treated:
            post_metrics += treatment_shift
        metrics = np.concatenate([pre_metrics, post_metrics])
        timestamps = np.concatenate([pre_period, post_period])
        return pd.DataFrame({"timestamp": timestamps, "unit": unit_name, "metric": metrics})
    
    treated_data = pd.concat([generate_unit_data(unit, is_treated=True) for unit in treatment_units])
    control_data = pd.concat([generate_unit_data(unit) for unit in control_units])
    df = pd.concat([treated_data, control_data]).reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).sort_values(by='timestamp').reset_index(drop=True)
    
    for unit in df.unit.unique():
        shift = shifts.get(unit, 0)
        n_rows = df[df.unit == unit].shape[0]
        df.loc[df.unit == unit, 'metric'] += shift + np.random.normal(0, random_noise_scale, n_rows)
    
    return df
