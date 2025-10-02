import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.tsa.stattools as stattools
import os

# Load the dataset Solar
timesteps = 4      # Sequence length abest value for 4 lags. It is also from autocorrelate
data_file_path = 'SolarEdge_Tageswerte_Modified.csv'
time_column = 'Time'
target_column= 'Inv 1 AC-Leistung (W)'
train_data_split = 0.9
train_data_path = "data/train_data_energy.csv"
eval_data_path = "data/eval_data.csv" #for energy data
#eval_data_labeled_path = "data/Labeled_SolarEdge_Eval_Dataset-original.csv"
eval_data_labeled_path = "data/synthetic_data_anomalies_energy.csv" # synthetic data for evaluation
anomaly_trashold = 85
window_size=7 #the whole week



# Load the dataset Machine
#data_file_path = 'data/machine_temperature_system_failure.csv'
#time_column = 'timestamp'#'Time'
#target_column= 'value'
#train_data_split = 0.9999
#train_data_path = "data/train_data.csv"
#eval_data_path = "data/train_data_not_cleaned_with_anomaly.csv"
#eval_data_labeled_path = "data/synthetic_data_anomalies.csv"
#anomaly_trashold = 85 # anomaly trashold 95 stands for high precision. If the requirement is to have high recall this number should go to range etween 80 and 90. 
#Add moving window features (e.g., 7-day window) or 24 for day in hourly, 288 for 5 min in a day
#window_size=288 #the whole day. Sensor measurament every 5 min meaning 288 per day

#is needed for autoencoder
def create_sequences(data, timesteps):
    """
    Converts a 2D array (samples, features) into a 3D array (samples, timesteps, features)
    using a sliding window.
    """
    data = np.asarray(data, dtype=np.float32)
    sequences = []
    for i in range(len(data) - timesteps + 1):
        sequences.append(data[i:i+timesteps])
    return np.array(sequences, dtype=np.float32)

    
#is needed for lstm
def create_forecasting_sequences(data, timesteps):
    """
    Converts a 2D array (samples, features) into input sequences (X) and targets (y)
    for one-step ahead forecasting.
    
    Parameters:
        data (np.array): Array of shape (n_samples, n_features)
        timesteps (int): Number of time steps in each input sequence.
        
    Returns:
        X (np.array): Array of shape (n_sequences, timesteps, n_features)
        y (np.array): Array of shape (n_sequences, n_features)
    """
    data = np.asarray(data, dtype=np.float32)
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i+timesteps])
        y.append(data[i+timesteps])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# Function to load and preprocess data
def load_data(file_path, time_column=None, drop_columns=None):
    """
    Loads data from a CSV file and performs basic preprocessing like handling time columns.

    Parameters:
    file_path (str): Path to the CSV file.
    time_column (str): Name of the time column to parse, if any.
    drop_columns (list): List of columns to drop from the dataset.

    Returns:
    pd.DataFrame: Preprocessed DataFrame.
    """
    data = pd.read_csv(file_path)
    if time_column:
        data[time_column] = pd.to_datetime(data[time_column], errors='coerce')
    if drop_columns:
        data = data.drop(columns=drop_columns)

        
    # Step 2: Remove 'W' from 'Inv 1 AC-Leistung (W)' and convert to numeric
    if 'Inv 1 AC-Leistung (W)' in data.columns and data['Inv 1 AC-Leistung (W)'].dtype == object: # if statement added. Be carefull
        data['Inv 1 AC-Leistung (W)'] = pd.to_numeric(data['Inv 1 AC-Leistung (W)'].str.replace('W', '', regex=False), errors='coerce')


    return data




def add_advanced_time_features(data, time_column):
    """
    Adds advanced time-based features extracted from a datetime column.
    
    This version preserves all rows, even if the datetime conversion fails.
    For rows with unparseable dates, the new features will be NaN.
    
    Parameters:
    data (pd.DataFrame): The dataset containing a datetime column.
    time_column (str): The name of the datetime column to extract features from.
    
    Returns:
    pd.DataFrame: Dataset with added time-based features.
    """
    data = data.copy()  # Work on a copy to avoid modifying the original DataFrame

    if time_column not in data.columns:
        raise KeyError(f"Column '{time_column}' not found in the DataFrame.")
    
    # Convert to datetime using dayfirst=True, but do not drop rows on errors.
    #deprecated since now i have date time already as format
    data[time_column] = pd.to_datetime(data[time_column], errors='coerce')
    
    # Extract basic time features; for unparseable dates, these will result in NaN.
    data['hour'] = data[time_column].dt.hour
    data['day'] = data[time_column].dt.day
    
    # Extract advanced time-based features
    data['day_of_week'] = data[time_column].dt.dayofweek  # Monday=0, Sunday=6
    
    # For the week of year, using isocalendar() (pandas >=1.1.0)
    try:
        data['week_of_year'] = data[time_column].dt.isocalendar().week
    except AttributeError:
        # Fallback for older versions of pandas (dt.week is deprecated in newer versions)
        data['week_of_year'] = data[time_column].dt.week
    
    data['month'] = data[time_column].dt.month
    
    # Binary feature for weekend (1 if Saturday or Sunday, else 0).
    data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if pd.notnull(x) and x >= 5 else 0)

    # this is added and possible reason for problems later on
    # Add cyclical hour features if available
    if 'hour' in data.columns:
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24.0)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24.0)
    
    return data


def add_lag_features(data, target_column, n_lags=3):
    """
    Adds lag features to the dataset for a specified target column.

    Parameters:
    data (pd.DataFrame): The dataset to add lag features to.
    target_column (str): The column name for which to create lagged features.
    n_lags (int): The number of lag features to add. Default is 3.

    Returns:
    pd.DataFrame: The dataset with added lag features.
    """
    # Add lag features for the specified number of lags
    for lag in range(1, n_lags + 1):
        data[f'lag_{lag}'] = data[target_column].shift(lag)
    
    # Drop rows with NaN values introduced by lagging
    data.dropna(inplace=True)
    
    return data


def add_moving_window_features(data, target_column, window_size=7):
    """
    Adds moving average and standard deviation features to the dataset.

    Parameters:
    - data (pd.DataFrame): The dataset to which the rolling statistics will be added.
    - target_column (str): The column on which to compute the rolling statistics.
    - window_size (int): The window size (in days or time steps) for the rolling calculations.

    Returns:
    - pd.DataFrame: The dataset with added moving average and moving std columns.
    """
    data = data.copy()
    # Compute rolling mean with min_periods=1
    data[f'moving_mean_{window_size}'] = (
        data[target_column]
        .rolling(window=window_size, min_periods=1)
        .mean()
    )
    # Compute rolling std with ddof=0 to avoid NaN for single observations
    data[f'moving_std_{window_size}'] = (
        data[target_column]
        .rolling(window=window_size, min_periods=1)
        .std(ddof=0)
    )
    return data

#for  data without scaling 
#data_preprocessed = preprocess_data_without_scaling(
#    eval_data_path, time_column, target_column, timesteps, drop_time_column=True
#)
def preprocess_data_without_scaling(file_path, time_column, target_column, timesteps, drop_time_column=False):
    """
    Preprocesses time series data by loading, cleaning, and adding features 
    such as lag features and moving window statistics.

    Steps performed:
      1. Loads data from the given path.
      2. Displays NaN counts before preprocessing.
      3. Handles missing values using interpolation and forward fill.
      4. Applies advanced time feature engineering.
      5. Adds lag features for the target column.
      6. Adds moving window features (mean and std).
      7. Optionally drops the 'Time' column (for evaluation datasets).

    Parameters:
      file_path (str): Path to the data CSV file.
      time_column (str): Name of the column containing datetime values.
      target_column (str): Name of the target column (used for creating lag features).
      timesteps (int): Number of lag steps to add.
      drop_time_column (bool): If True, drops the 'Time' column (for evaluation datasets).

    Returns:
      pd.DataFrame: Preprocessed dataset ready for modeling.
    """

    # Step 1: Load the dataset
    data = load_data(file_path)
    
    # Step 2: Display NaN counts before preprocessing
    print("NaN counts before preprocessing:")
    print(data.isna().sum())
     # Infer proper dtypes before interpolation
    data.infer_objects(copy=False)
    
    # Step 3: Handle missing values using interpolation and forward fill
    data.interpolate(method='linear', inplace=True)
    data.ffill(inplace=True)

    # Step 4: Apply advanced time features
    data = add_advanced_time_features(data, time_column)

    # Step 5: Display NaN counts after adding time features
    print("NaN counts after adding time features:")
    print(data.isna().sum())

    # Step 6: Add lag features for the target column
    data = add_lag_features(data, target_column, n_lags=timesteps)

    # Step 7: Add moving window features (e.g., 7-day window) or 24 for day in hourly, 288 for 5 min in a day
    data = add_moving_window_features(data, target_column, window_size)

    # Step 8: Drop 'Time' column if needed (for evaluation data)
    if drop_time_column and time_column in data.columns:
        data.drop(columns=[time_column], inplace=True)

    return data

# fpr training data 
#train_data_scaled = preprocess_data_with_scaling(
#    train_data_path, time_column, target_column, timesteps, scaler, drop_time_column=False
#)
def preprocess_data_with_scaling(file_path, time_column, target_column, timesteps, scaler, drop_time_column=False):
    """
    Preprocesses time series data by calling `preprocess_data_without_scaling`, 
    then scales the numeric features.

    Steps performed:
      1. Calls `preprocess_data_without_scaling()` to perform:
         - Loading data
         - Handling missing values
         - Applying advanced time feature engineering
         - Adding lag features
         - Adding moving window features (mean and std)
      2. Selects feature columns (excluding specified ones).
      3. Scales the selected numeric columns.
      4. Optionally drops the 'Time' column.

    Parameters:
      file_path (str): Path to the data CSV file.
      time_column (str): Name of the column containing datetime values.
      target_column (str): Name of the target column (used for creating lag features).
      timesteps (int): Number of lag steps to add.
      scaler (object): A fitted scaler (or a scaler instance to be fitted) such as StandardScaler.
      drop_time_column (bool): If True, drops the 'Time' column (for evaluation datasets).

    Returns:
      pd.DataFrame: Preprocessed, scaled numeric evaluation data ready for model input.
    """

    # Step 1: Perform initial preprocessing without scaling
    data = preprocess_data_without_scaling(file_path, time_column, target_column, timesteps, drop_time_column)

    # Step 2: Prepare feature columns for scaling
    feature_columns = data.columns.tolist()
    exclude_columns = ['hour','hour_cos', 'hour_sin', 'day', 'day_of_week', 'week_of_year', 'month', 'is_weekend', 'anomaly_label', time_column]
    feature_columns_to_scale = [col for col in feature_columns if col not in exclude_columns]

    # Step 3: Scale the selected columns using a custom scaling function.
    data = scale_selected_columns(data, feature_columns_to_scale, scaler)

    return data





#===================================================================================


def scale_data(scaler, data, target_column, n_lags):
    """
    Scales the target column, lag columns, and moving window features using StandardScaler.

    Parameters:
    scaler (StandardScaler): Scaler instance (e.g., StandardScaler, MinMaxScaler) used to scale the data.
    data (pd.DataFrame): The dataset containing the data.
    target_column (str): The main column to be normalized/scaled (e.g., 'Inv 1 AC-Leistung (W)').
    n_lags (int): The number of lag features to add and scale dynamically.

    Returns:
    pd.DataFrame: The dataset with scaled target column, lag columns, and moving window features.
    list: List of the columns that were scaled.
    """

    # Add lag features dynamically based on n_lags
    #data_with_lags = add_lag_features(data, target_column, n_lags)

    # Dynamically create a list of columns to scale: target column, lag columns, and moving window features
    columns_to_scale = [target_column] + [f'lag_{i}' for i in range(1, n_lags + 1)]

    # Check if moving window features exist in the dataset before adding them
    moving_window_features = [f'moving_mean_{window_size}', f'moving_std_{window_size}']
    existing_moving_features = [col for col in moving_window_features if col in data.columns]
    
    # Add moving window features if they exist
    columns_to_scale.extend(existing_moving_features)

    # Ensure no NaN values in the columns to be scaled
    if data[columns_to_scale].isna().any().any():
        raise ValueError(f"Missing values found in columns: {columns_to_scale}")

    # Scale the selected columns
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

    return data, columns_to_scale



def scale_selected_columns(data, feature_columns_to_scale, scaler):
    """
    Scales only the specified columns in the dataset using the provided scaler.

    Parameters:
    - data (pd.DataFrame): The dataset with features to scale.
    - feature_columns_to_scale (list): List of column names to be scaled.
    - scaler (object): An instance of a scaler (e.g., StandardScaler or MinMaxScaler).

    Returns:
    - data (pd.DataFrame): The dataset with only the specified columns scaled.
    """
    # Copy the dataset to avoid modifying the original data
    data_scaled = data.copy()

    # Apply the scaler to the selected columns
    data_scaled[feature_columns_to_scale] = scaler.transform(data[feature_columns_to_scale])

    return data_scaled
    
#deprecated
def filter_data_by_date(data, time_column, max_date_str, days_offset=200, n_years_for_training=2):
    """
    Filters the data based on a specified date range.
    
    Parameters:
    data (pd.DataFrame): The dataset containing the data.
    time_column (str): The column containing datetime values (e.g., 'Time').
    max_date_str (str): The maximum date for analysis as a string (format: 'YYYY-MM-DD').
    days_offset (int): The number of days offset for the training size. Default is 200.
    n_years_for_training (int): The number of years for the training period. Default is 2 years.
    
    Returns:
    pd.DataFrame: The filtered dataset based on the date range.
    """
    
    # Convert max_date to a timestamp
    max_date = pd.Timestamp(max_date_str)
    
    # Calculate date offsets for training and testing ranges
    max_date_minus_n_years = max_date - pd.DateOffset(years=n_years_for_training)
    max_date_minus_1_year = max_date - pd.DateOffset(years=1)
    max_date_minus_100 = max_date - pd.DateOffset(days=days_offset)  # Example: 200 days before max_date
    
    # Filter the data for the training period (from max_date_minus_n_years to max_date_minus_1_year)
    train_data = data[(data[time_column] >= max_date_minus_n_years) & 
                      (data[time_column] < max_date_minus_1_year)]

    return train_data

# import the initial data set and split this data into train and eval data sets

def filter_and_save_data(input_file, time_column, training_percentage, training_file, eval_file):
    """
    Reads the input file, splits the data into training and evaluation sets 
    based on the specified percentage, and saves the resulting sets in the 
    same format as the input file.

    Parameters:
    - input_file (str): Path to the input data file (e.g., CSV or Excel).
    - time_column (str): The column containing datetime values.
    - training_percentage (float): Fraction of data to use for training 
                                   (e.g., 0.7 for 70%).
    - training_file (str): Path to save the training set.
    - eval_file (str): Path to save the evaluation set.
    """
    # Determine file extension to decide read/write method
    _, ext = os.path.splitext(input_file)
    ext = ext.lower()
    
    # Read the data based on file extension
    if ext == '.csv':
        data = pd.read_csv(input_file)
    elif ext in ['.xls', '.xlsx']:
        data = pd.read_excel(input_file)
    else:
        raise ValueError("Unsupported file format: {}".format(ext))
    
    # Try parsing with DD.MM.YYYY format first
    try:
        data[time_column] = pd.to_datetime(data[time_column], format='%d.%m.%Y')
    except ValueError:
        # Fallback: automatically infer the datetime format 
        # (handles e.g. 2013-12-02 21:15:00 or 2019-12-04)
        data[time_column] = pd.to_datetime(
            data[time_column], 
            infer_datetime_format=True, 
            errors='coerce'
        )
    
    # Sort the data in ascending order based on the time column
    data_sorted = data.sort_values(by=time_column)
    
    # Calculate the number of samples for training
    n_train = int(len(data_sorted) * training_percentage)
    
    # Split the data into training and evaluation sets
    train_data = data_sorted.iloc[:n_train]
    eval_data = data_sorted.iloc[n_train:]
    
    # Save the data in the same format as the input file
    if ext == '.csv':
        train_data.to_csv(training_file, index=False)
        eval_data.to_csv(eval_file, index=False)
    elif ext in ['.xls', '.xlsx']:
        with pd.ExcelWriter(training_file) as writer:
            train_data.to_excel(writer, index=False, sheet_name='Training')
        with pd.ExcelWriter(eval_file) as writer:
            eval_data.to_excel(writer, index=False, sheet_name='Evaluation')
    else:
        # Should not reach here because of earlier check.
        raise ValueError("Unsupported file format for saving: {}".format(ext))
    
    print("Data successfully split and saved to:\n  Training: {}\n  Evaluation: {}".format(training_file, eval_file))

# Example usage:
# filter_and_save_data("data.csv", "Time", 0.7, "train_data.csv", "eval_data.csv")

    
#=====================================================================================================



# Function to filter data for the last X days based on a specified max date and offset

def filter_last_x_days(data, time_column, max_date_str, days_offset=200):
    """
    Filters data for the last X days based on a specified max date and offset.

    Parameters:
    data (pd.DataFrame): The dataset containing the data to be filtered.
    time_column (str): The column name that contains the datetime values.
    max_date_str (str): The max date for the filter in 'YYYY-MM-DD' format.
    days_offset (int): The number of days before the max_date to start the filtering.

    Returns:
    pd.DataFrame: The filtered dataset containing only the rows within the date range.
    """
    # Convert max_date to a timestamp
    max_date = pd.Timestamp(max_date_str)

    # Calculate the date offset
    max_date_minus_offset = max_date - pd.DateOffset(days=days_offset)

    # Ensure the time column is in datetime format
    data[time_column] = pd.to_datetime(data[time_column], errors='coerce')

    # Filter the data based on the date range
    filtered_data = data[(data[time_column] > max_date_minus_offset) & (data[time_column] <= max_date)]

    return filtered_data



# Function to visualize anomalies on the time series
def plot_anomalies(data, time_column, value_column, anomaly_column):
    """
    Plots the time series and highlights the detected anomalies.
    
    Parameters:
    data (pd.DataFrame): The dataset containing time, values, and anomaly columns.
    time_column (str): The name of the time column.
    value_column (str): The name of the column containing the values (e.g., power output).
    anomaly_column (str): The name of the column containing anomaly labels (1 for anomaly, 0 for normal).
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot the value over time
    ax.plot(data[time_column], data[value_column], label='Power Output (W)', color='blue')

    # Highlight anomalies
    anomalies = data[data[anomaly_column] == 1]
    ax.scatter(anomalies[time_column], anomalies[value_column], color='red', label='Anomalies')

    # Labels and title
    ax.set_title('Power Output with Anomalies Highlighted')
    ax.set_xlabel('Time')
    ax.set_ylabel('Power Output (W)')
    plt.legend()
    plt.show()

