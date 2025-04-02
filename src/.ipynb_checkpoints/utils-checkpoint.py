import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import nnls

#-----------------------------------------------------------------------------------------
#   FUNCTIONS FOR DATA VIZ
#-----------------------------------------------------------------------------------------

def plot_timeseries(df, time_col, unit_col, metric_col, treatment_units, treatment_timestamp=None, 
                    figsize=(8, 4), tick_rotation=None):
    """
    Plots the timeseries for each unit, highlighting the treatment units.

    Parameters:
    - df: pandas DataFrame containing the data.
    - time_col: str, the name of the time column.
    - unit_col: str, the name of the unit column.
    - metric_col: str, the name of the metric column to be analyzed.
    - treatment_units: list, units that received the treatment.
    - treatment_timestamp: Timestamp, the first timestamp of the treatment period.
    - figsize: tuple, the size of the figure.
    - tick_rotation: angle of rotation of timestamp labels.
    """
    df[time_col] = pd.to_datetime(df[time_col])
    fig, ax = plt.subplots(figsize=figsize)
    units = df[unit_col].unique()
    units = sorted(units, key=lambda x: x in treatment_units, reverse=True)

    for unit in units:
        unit_data = df[df[unit_col] == unit]
        if unit in treatment_units:
            ax.plot(unit_data[time_col], unit_data[metric_col], label=unit, linewidth=2.5)
        else:
            ax.plot(unit_data[time_col], unit_data[metric_col], label=unit, alpha=0.9)
    
    if treatment_timestamp:
        treatment_timestamp = pd.Timestamp(treatment_timestamp)
        ax.axvline(x=treatment_timestamp, color='black', linestyle='--', label='Treatment Start')

    ax.set(xlabel=time_col, ylabel=metric_col)
    if tick_rotation: ax.tick_params(axis='x', rotation=tick_rotation)
    ax.legend()
    ax.grid(True, alpha=0.5, linestyle='--')
    plt.show()

    
    
#-----------------------------------------------------------------------------------------
#   FUNCTIONS FOR DID
#   TODO: check the function that also returns lift
#-----------------------------------------------------------------------------------------


def convert_bool_to_int(df):
    """
    Convert all boolean columns in a DataFrame to integers.

    Parameters:
    - df: pandas DataFrame containing the data.

    Returns:
    - df: pandas DataFrame with boolean columns converted to integers.
    """
    bool_cols = df.select_dtypes(include=[bool]).columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def prepare_fixed_effects_data(df, time_col, unit_col, metric_col, treatment_timestamp, treatment_units):
    """
    Prepare the data for a fixed effects regression using statsmodels.

    Parameters:
    - df: pandas DataFrame containing the data.
    - time_col: str, the name of the time column.
    - unit_col: str, the name of the unit column.
    - metric_col: str, the name of the metric column to be analyzed.
    - treatment_timestamp: Timestamp, the timestamp indicating the treatment period.
    - treatment_units: list, units that received the treatment.

    Returns:
    - prepared_df: pandas DataFrame prepared for fixed effects regression.
    """
    df[time_col] = pd.to_datetime(df[time_col])
    df['After'] = df[time_col] >= treatment_timestamp
    df['Treated'] = df[unit_col].isin(treatment_units) & df['After']
    df['original_unit'] = df[unit_col]
    df = pd.get_dummies(df, columns=[unit_col, time_col], drop_first=True)
    df.columns = df.columns.str.replace('-', '_').str.replace(':', '_').str.replace(' ', '_')
    fixed_effects_cols = [col for col in df.columns if col.startswith('unit_') or col.startswith(time_col + '_')]
    prepared_df = df[[metric_col, 'Treated'] + fixed_effects_cols + ['original_unit']]
    prepared_df = convert_bool_to_int(prepared_df)
    return prepared_df


def fit_fixed_effects_model(df_fe, metric_col, time_col, clustered_errors=False, model_summary=False):
    """
    Fit a fixed effects regression model using statsmodels.

    Parameters:
    - df_fe: pandas DataFrame prepared for fixed effects regression.
    - metric_col: str, the name of the metric column to be analyzed.
    - clustered_errors: boolean, whether to use clustered standard errors.
    - model_summary: boolean, whether to print the model summary.

    Returns:
    - did_coefficient: float, the coefficient of the DID estimate.
    - p_value: float, the p-value of the DID estimate.
    - confidence_interval: tuple, the confidence interval of the DID estimate.
    """
    fixed_effects_terms = ' + '.join([col for col in df_fe.columns if col.startswith('unit_') or col.startswith(time_col+'_')])
    formula = f'{metric_col} ~ Treated + {fixed_effects_terms}'
    if clustered_errors:
        model = smf.ols(formula, data=df_fe).fit(cov_type='cluster', cov_kwds={'groups': df_fe['original_unit']})
    else:
        model = smf.ols(formula, data=df_fe).fit()
    if model_summary:
        print(model.summary())
    did_coefficient = model.params['Treated']
    p_value = model.pvalues['Treated']
    confidence_interval = model.conf_int().loc['Treated'].values
        
    return did_coefficient, p_value, confidence_interval, model

def fit_fixed_effects_model_with_lift(df_fe, metric_col, time_col, unit_col, treatment_units, treatment_timestamp, clustered_errors=False, model_summary=False):
    """
    Fit a fixed effects regression model using statsmodels and compute the lift.

    Parameters:
    - df_fe: pandas DataFrame prepared for fixed effects regression.
    - metric_col: str, the name of the metric column to be analyzed.
    - time_col: str, the name of the time column.
    - unit_col: str, the name of the unit column.
    - treatment_units: list, the treated units.
    - treatment_timestamp: Timestamp, when the treatment starts.
    - clustered_errors: boolean, whether to use clustered standard errors.
    - model_summary: boolean, whether to print the model summary.

    Returns:
    - did_coefficient: float, the coefficient of the DID estimate.
    - p_value: float, the p-value of the DID estimate.
    - confidence_interval: tuple, the confidence interval of the DID estimate.
    - lift: float, the relative effect size compared to the counterfactual.
    - model: statsmodels regression results object.
    """
    # Define the regression formula
    fixed_effects_terms = ' + '.join([col for col in df_fe.columns if col.startswith('unit_') or col.startswith(time_col+'_')])
    formula = f'{metric_col} ~ Treated + {fixed_effects_terms}'

    # Fit the model
    if clustered_errors:
        model = smf.ols(formula, data=df_fe).fit(cov_type='cluster', cov_kwds={'groups': df_fe[unit_col]})
    else:
        model = smf.ols(formula, data=df_fe).fit()

    if model_summary:
        print(model.summary())

    # Extract DID results
    did_coefficient = model.params['Treated']
    p_value = model.pvalues['Treated']
    confidence_interval = model.conf_int().loc['Treated'].values

    # Compute counterfactual using trend adjustment
    pre_treatment_avg_treatment = df_fe[
        (df_fe[time_col] < treatment_timestamp) & (df_fe[unit_col].isin(treatment_units))
    ][metric_col].mean()

    pre_treatment_avg_control = df_fe[
        (df_fe[time_col] < treatment_timestamp) & (~df_fe[unit_col].isin(treatment_units))
    ][metric_col].mean()

    post_treatment_avg_control = df_fe[
        (df_fe[time_col] >= treatment_timestamp) & (~df_fe[unit_col].isin(treatment_units))
    ][metric_col].mean()

    trend_adjustment = post_treatment_avg_control - pre_treatment_avg_control
    counterfactual = pre_treatment_avg_treatment + trend_adjustment

    # Compute lift
    lift = did_coefficient / counterfactual if counterfactual != 0 else None

    return did_coefficient, p_value, confidence_interval, lift, model


#-----------------------------------------------------------------------------------------
#   FUNCTIONS FOR SYNTHETIC DID
#   TODO: check the function that also returns lift
#-----------------------------------------------------------------------------------------

from scipy.optimize import nnls

def create_synthetic_control(df, time_col, unit_col, metric_col, treatment_units, treatment_timestamp):
    """
    Create synthetic control units by finding the weights for the control units that best match
    the pre-treatment outcomes of each treated unit.

    Parameters:
    - df: pandas DataFrame containing the data.
    - time_col: str, the name of the time column.
    - unit_col: str, the name of the unit column.
    - metric_col: str, the name of the metric column to be analyzed.
    - treatment_units: list, the units that received the treatment.
    - treatment_timestamp: Timestamp, the first timestamp of the treatment period.

    Returns:
    - synthetic_controls: dict, mapping each treatment unit to its synthetic control DataFrame.
    - weights_dict: dict, mapping each treatment unit to the weights for control units.
    """
    # Ensure treatment_units is a list
    if isinstance(treatment_units, str):
        treatment_units = [treatment_units]
    
    # Ensure the time column is in datetime format
    df[time_col] = pd.to_datetime(df[time_col])

    # Filter the data for the pre-treatment period
    pre_treatment_df = df[df[time_col] < treatment_timestamp]
    
    # Get all control units (units not in treatment_units)
    control_unit_names = [unit for unit in df[unit_col].unique() if unit not in treatment_units]
    
    # Initialize dictionaries to store results
    synthetic_controls = {}
    weights_dict = {}
    
    # For each treatment unit, create a synthetic control
    for treatment_unit in treatment_units:
        # Get the treated unit data
        treated_unit_data = pre_treatment_df[pre_treatment_df[unit_col] == treatment_unit]
        
        if len(treated_unit_data) == 0:
            print(f"Warning: No pre-treatment data found for unit {treatment_unit}")
            continue
            
        # Get the control units data
        control_units = pre_treatment_df[pre_treatment_df[unit_col].isin(control_unit_names)]
        
        # Create pivot table for control units
        control_units_pivot = control_units.pivot(index=time_col, columns=unit_col, values=metric_col)
        
        # Ensure the control units pivot table has the same time points as the treated unit
        common_timepoints = sorted(set(treated_unit_data[time_col]).intersection(set(control_units_pivot.index)))
        
        if len(common_timepoints) == 0:
            print(f"Warning: No common timepoints between treatment unit {treatment_unit} and control units")
            continue
            
        # Filter to common timepoints
        control_units_pivot = control_units_pivot.loc[common_timepoints]
        treated_unit_outcomes = treated_unit_data.set_index(time_col).loc[common_timepoints, metric_col]
        
        # Drop any columns with NaN values
        control_units_pivot = control_units_pivot.dropna(axis=1)
        
        # Solve for the weights using non-negative least squares
        X = control_units_pivot.values
        y = treated_unit_outcomes.values
        
        try:
            weights, _ = nnls(X, y)
            
            # Create weight dictionary mapping control unit names to weights
            weight_map = dict(zip(control_units_pivot.columns, weights))
            weights_dict[treatment_unit] = weight_map
            
            # Create the synthetic control unit
            synthetic_control = control_units_pivot @ weights
            synthetic_control = pd.DataFrame({
                time_col: control_units_pivot.index, 
                metric_col: synthetic_control,
                unit_col: f"Synthetic_{treatment_unit}"
            })
            
            # Store the synthetic control
            synthetic_controls[treatment_unit] = synthetic_control
            
        except Exception as e:
            print(f"Error creating synthetic control for unit {treatment_unit}: {e}")
    
    return synthetic_controls, weights_dict


def estimate_synthetic_did(df, time_col, unit_col, metric_col, treatment_units, treatment_timestamp, 
                          clustered_errors=False, model_summary=False):
    """
    Estimate the synthetic DID effect by comparing the post-treatment outcomes of treated units 
    to their synthetic control units.

    Parameters:
    - df: pandas DataFrame containing the data.
    - time_col: str, the name of the time column.
    - unit_col: str, the name of the unit column.
    - metric_col: str, the name of the metric column to be analyzed.
    - treatment_units: list or str, the unit(s) that received the treatment.
    - treatment_timestamp: Timestamp, the first timestamp of the treatment period.
    - clustered_errors: boolean, whether to use clustered standard errors.
    - model_summary: boolean, whether to print the model summary.

    Returns:
    - did_coefficient: float, the coefficient of the DID estimate.
    - p_value: float, the p-value of the DID estimate.
    - confidence_interval: tuple, the confidence interval of the DID estimate.
    - model: statsmodels regression results object.
    """
    # Ensure treatment_units is a list
    if isinstance(treatment_units, str):
        treatment_units = [treatment_units]
    
    # Ensure the time column is in datetime format
    df[time_col] = pd.to_datetime(df[time_col])

    # Create synthetic control units
    synthetic_controls, weights_dict = create_synthetic_control(
        df, time_col, unit_col, metric_col, treatment_units, treatment_timestamp
    )
    
    if len(synthetic_controls) == 0:
        raise ValueError("Could not create synthetic controls for any treatment units")
    
    # Combine all synthetic controls into one dataframe
    synthetic_df = pd.concat([sc for sc in synthetic_controls.values()], ignore_index=True)
    
    # Combine with original data
    # First, identify which rows in the original df to keep (exclude treatment units)
    orig_df_to_keep = df[~df[unit_col].isin(treatment_units)].copy()
    
    # Add the actual treated units
    treated_df = df[df[unit_col].isin(treatment_units)].copy()
    
    # Combine everything
    combined_df = pd.concat([orig_df_to_keep, treated_df, synthetic_df], ignore_index=True)
    
    # Create a binary treatment indicator
    combined_df['Is_Treated'] = combined_df[unit_col].isin(treatment_units)
    combined_df['Is_Post'] = combined_df[time_col] >= treatment_timestamp
    combined_df['Treated_X_Post'] = combined_df['Is_Treated'] & combined_df['Is_Post']
    
    # Create unit and time fixed effects
    unit_dummies = pd.get_dummies(combined_df[unit_col], prefix='unit', drop_first=True)
    time_dummies = pd.get_dummies(combined_df[time_col], prefix='time', drop_first=True)
    
    # Combine all variables for regression
    X = pd.concat([
        combined_df[['Treated_X_Post']],
        unit_dummies,
        time_dummies
    ], axis=1).astype(float)
    
    # Add constant
    X = sm.add_constant(X)
    
    # Fit model
    y = combined_df[metric_col].astype(float)
    
    if clustered_errors:
        model = sm.OLS(y, X).fit(
            cov_type='cluster', 
            cov_kwds={'groups': combined_df[unit_col]}
        )
    else:
        model = sm.OLS(y, X).fit()
    
    if model_summary:
        print(model.summary())
    
    # Extract results
    did_coefficient = model.params['Treated_X_Post']
    p_value = model.pvalues['Treated_X_Post']
    confidence_interval = model.conf_int().loc['Treated_X_Post'].values
    
    return did_coefficient, p_value, confidence_interval, model, weights_dict


def estimate_synthetic_did_with_lift(df, time_col, unit_col, metric_col, treatment_units, treatment_timestamp, 
                          clustered_errors=False, model_summary=False):
    """
    Estimate the synthetic DID effect by comparing the post-treatment outcomes of treated units 
    to their synthetic control units.

    Parameters:
    - df: pandas DataFrame containing the data.
    - time_col: str, the name of the time column.
    - unit_col: str, the name of the unit column.
    - metric_col: str, the name of the metric column to be analyzed.
    - treatment_units: list or str, the unit(s) that received the treatment.
    - treatment_timestamp: Timestamp, the first timestamp of the treatment period.
    - clustered_errors: boolean, whether to use clustered standard errors.
    - model_summary: boolean, whether to print the model summary.

    Returns:
    - did_coefficient: float, the coefficient of the DID estimate.
    - p_value: float, the p-value of the DID estimate.
    - confidence_interval: tuple, the confidence interval of the DID estimate.
    - lift: float, the relative effect (did_coefficient divided by counterfactual).
    - counterfactual: float, the expected value for treated units in post period if no treatment.
    - model: statsmodels regression results object.
    - weights_dict: dict, mapping of control weights for each treatment unit.
    """
    # Ensure treatment_units is a list
    if isinstance(treatment_units, str):
        treatment_units = [treatment_units]
    
    # Ensure the time column is in datetime format
    df[time_col] = pd.to_datetime(df[time_col])

    # Create synthetic control units
    synthetic_controls, weights_dict = create_synthetic_control(
        df, time_col, unit_col, metric_col, treatment_units, treatment_timestamp
    )
    
    if len(synthetic_controls) == 0:
        raise ValueError("Could not create synthetic controls for any treatment units")
    
    # Combine all synthetic controls into one dataframe
    synthetic_df = pd.concat([sc for sc in synthetic_controls.values()], ignore_index=True)
    
    # Create a complete synthetic control dataframe for all time periods
    all_time_periods = df[time_col].unique()
    synthetic_complete = pd.DataFrame()
    
    for unit, weights in weights_dict.items():
        # Get control unit data for all time periods
        control_data = df[df[unit_col].isin(weights.keys())]
        
        # For each time period, calculate the weighted average of control units
        for time in all_time_periods:
            time_data = control_data[control_data[time_col] == time]
            if len(time_data) > 0:
                weighted_sum = 0
                weight_sum = 0
                for control_unit, weight in weights.items():
                    if control_unit in time_data[unit_col].values:
                        unit_value = time_data[time_data[unit_col] == control_unit][metric_col].values[0]
                        weighted_sum += unit_value * weight
                        weight_sum += weight
                
                if weight_sum > 0:
                    synthetic_value = weighted_sum / weight_sum
                    synthetic_complete = pd.concat([
                        synthetic_complete,
                        pd.DataFrame({
                            time_col: [time],
                            unit_col: [f"Synthetic_{unit}"],
                            metric_col: [synthetic_value]
                        })
                    ], ignore_index=True)
    
    # Use this more complete synthetic control data if available
    if len(synthetic_complete) > 0:
        synthetic_df = synthetic_complete
    
    # Combine with original data
    # First, identify which rows in the original df to keep (exclude treatment units)
    orig_df_to_keep = df[~df[unit_col].isin(treatment_units)].copy()
    
    # Add the actual treated units
    treated_df = df[df[unit_col].isin(treatment_units)].copy()
    
    # Combine everything
    combined_df = pd.concat([orig_df_to_keep, treated_df, synthetic_df], ignore_index=True)
    
    # Create a binary treatment indicator
    combined_df['Is_Treated'] = combined_df[unit_col].isin(treatment_units)
    combined_df['Is_Post'] = combined_df[time_col] >= treatment_timestamp
    combined_df['Treated_X_Post'] = combined_df['Is_Treated'] & combined_df['Is_Post']
    
    # Create unit and time fixed effects
    unit_dummies = pd.get_dummies(combined_df[unit_col], prefix='unit', drop_first=True)
    time_dummies = pd.get_dummies(combined_df[time_col], prefix='time', drop_first=True)
    
    # Combine all variables for regression
    X = pd.concat([
        combined_df[['Treated_X_Post']],
        unit_dummies,
        time_dummies
    ], axis=1).astype(float)
    
    # Add constant
    X = sm.add_constant(X)
    
    # Fit model
    y = combined_df[metric_col].astype(float)
    
    if clustered_errors:
        model = sm.OLS(y, X).fit(
            cov_type='cluster', 
            cov_kwds={'groups': combined_df[unit_col]}
        )
    else:
        model = sm.OLS(y, X).fit()
    
    if model_summary:
        print(model.summary())
    
    # Extract results
    did_coefficient = model.params['Treated_X_Post']
    p_value = model.pvalues['Treated_X_Post']
    confidence_interval = model.conf_int().loc['Treated_X_Post'].values
    
    # Calculate the counterfactual 
    # 1. Get the actual post-treatment average for treated units
    actual_post = df[(df[unit_col].isin(treatment_units)) & 
                    (df[time_col] >= treatment_timestamp)][metric_col].mean()
    
    # 2. Get the post-treatment average for synthetic controls (representing what would have happened)
    synthetic_post = synthetic_df[synthetic_df[time_col] >= treatment_timestamp][metric_col].mean()
    
    # 3. The counterfactual is what treated units would have been without treatment
    # This is the actual value minus the treatment effect
    counterfactual = actual_post - did_coefficient
    
    # Double-check with synthetic control value (should be similar)
    if abs(counterfactual - synthetic_post) > 0.1 * abs(synthetic_post):
        # If there's a large discrepancy, use the synthetic control value as it's more reliable
        counterfactual = synthetic_post
    
    # Calculate lift as percentage change
    if counterfactual != 0:
        lift = did_coefficient / counterfactual
    else:
        lift = None  # Can't calculate lift if counterfactual is zero
    
    return did_coefficient, p_value, confidence_interval, lift, counterfactual, model, weights_dict


#-----------------------------------------------------------------------------------------
#   THE FUNCTIONS BELOW ARE RELATED TO POWER ANALYSIS
#-----------------------------------------------------------------------------------------

def bootstrap_sample(df, metric_col, time_col, unit_col, n_bootstrap, time_initial):
    """
    Generates bootstrap samples of consecutive timestamps and aligns them with a new increasing timestamp.
    
    Parameters:
    - df (pd.DataFrame): The panel data.
    - metric_col (str): The name of the metric column.
    - time_col (str): The name of the time column.
    - unit_col (str): The name of the unit column.
    - n_bootstrap (int): The number of consecutive timestamps to bootstrap.
    - time_initial (pd.Timestamp): The starting timestamp for the bootstrapped data.

    Returns:
    - pd.DataFrame: Bootstrapped data with continuous time series.
    """
    df = df.sort_values(time_col)
    original_timestamps = df[time_col].unique()
    cadence = original_timestamps[1] - original_timestamps[0]  # Assumes uniform spacing
    bootstrapped_timestamps = np.random.choice(original_timestamps, size=n_bootstrap, replace=True)
    bootstrapped_df = pd.DataFrame(columns=df.columns)
    new_timestamps = [time_initial + i * cadence for i in range(n_bootstrap)]
    
    for i, timestamp in enumerate(bootstrapped_timestamps):
        slice_df = df[df[time_col] == timestamp].copy()
        slice_df[time_col] = new_timestamps[i]
        bootstrapped_df = pd.concat([bootstrapped_df, slice_df], ignore_index=True)
    
    return bootstrapped_df

def create_data(df, metric_col, time_col, unit_col, treatment_units, period_length, mde_pc):
    """
    Creates synthetic panel data with multiple treatments from bootstrapped samples.
    
    Parameters:
    - df (pd.DataFrame): The original panel data.
    - metric_col (str): The metric column name.
    - time_col (str): The time column name.
    - unit_col (str): The unit column name.
    - treatment_units (list): List of treatment units.
    - period_length (int): The number of days in pre- and post-treatment periods.
    - mde_pc (float): The minimum detectable effect as a percentage of the pre-period average.
    
    Returns:
    - pd.DataFrame: Synthetic panel data.
    - pd.Timestamp: Treatment start timestamp.
    """
    time_initial = df[time_col].min()
    synthetic_data = bootstrap_sample(df, metric_col, time_col, unit_col, n_bootstrap=2*period_length, time_initial=time_initial)
    pre_period = synthetic_data[time_col].unique()[:period_length]
    post_period = synthetic_data[time_col].unique()[period_length:]
    treatment_timestamp = post_period[0]
    
    for treatment_unit in treatment_units:
        pre_treatment_avg = synthetic_data[(synthetic_data[unit_col] == treatment_unit) & (synthetic_data[time_col].isin(pre_period))][metric_col].mean()
        mde = pre_treatment_avg * mde_pc
        synthetic_data.loc[(synthetic_data[unit_col] == treatment_unit) & (synthetic_data[time_col].isin(post_period)), metric_col] += mde
    
    return synthetic_data, treatment_timestamp

def est_model(df, metric_col, time_col, unit_col, treatment_timestamp, treatment_units, clustered_errors=True):
    """
    Estimates a fixed effects model on synthetic data with multiple treatments.
    
    Parameters:
    - df (pd.DataFrame): The synthetic panel data.
    - metric_col (str): The metric column.
    - time_col (str): The time column.
    - unit_col (str): The unit column.
    - treatment_timestamp (pd.Timestamp): Start of the treatment period.
    - treatment_units (list): List of treated units.
    - clustered_errors (bool, optional): Whether to use clustered errors. Default is True.
    
    Returns:
    - dict: A dictionary mapping each treatment unit to its p-value.
    """
    p_values = {}
    for treatment_unit in treatment_units:
        prepared_df = prepare_fixed_effects_data(df, time_col, unit_col, metric_col, treatment_timestamp, treatment_units)
        _, p_value, _, _ = fit_fixed_effects_model(prepared_df, metric_col, time_col, clustered_errors)
        p_values[treatment_unit] = p_value
    return p_values

def simulate_power(df, metric_col, time_col, unit_col, treatment_units, period_length, mde_pc, n_iterations, α, clustered_errors=True):
    """
    Simulates power by estimating model significance over multiple iterations with multiple treatments.
    
    Parameters:
    - df (pd.DataFrame): The original panel data.
    - metric_col (str): The metric column.
    - time_col (str): The time column.
    - unit_col (str): The unit column.
    - treatment_units (list): List of treated units.
    - period_length (int): Number of days in pre- and post-treatment periods.
    - mde_pc (float): Minimum detectable effect as a percentage.
    - n_iterations (int): Number of simulation iterations.
    - α (float): Significance level.
    - clustered_errors (bool, optional): Use clustered errors. Default is True.
    
    Returns:
    - dict: A dictionary mapping each treatment unit to its estimated power.
    """
    power_results = {unit: 0 for unit in treatment_units}
    
    for _ in range(n_iterations):
        synthetic_data, treatment_timestamp = create_data(df, metric_col, time_col, unit_col, treatment_units, period_length, mde_pc)
        p_values = est_model(synthetic_data, metric_col, time_col, unit_col, treatment_timestamp, treatment_units, clustered_errors)
        
        for unit, p_value in p_values.items():
            if p_value < α:
                power_results[unit] += 1
    
    for unit in power_results:
        power_results[unit] /= n_iterations
    
    return power_results
