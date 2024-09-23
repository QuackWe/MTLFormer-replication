import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import torch


def load_helpdesk():
    # Load Helpdesk CSV
    helpdesk_df = pd.read_csv('data/Helpdesk.csv')

    # Sorting by caseID and timestamp
    helpdesk_df = helpdesk_df.sort_values(by=['Case ID', 'Complete Timestamp'])

    # Grouping events by each case (process instance)
    helpdesk_traces = helpdesk_df.groupby('Case ID').agg(list)

    # Display one trace
    return helpdesk_df, helpdesk_traces


def calculate_time_features(df):
    df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'])  # Convert to datetime

    # Initialize time columns
    df['next_event_time'] = 0
    df['remaining_time'] = 0

    # Iterate through each case to calculate next event time and remaining time
    for case_id, group in df.groupby('Case ID'):
        timestamps = group['Complete Timestamp'].values
        num_events = len(timestamps)

        # Calculate next event time and remaining time for each event in the trace
        for i in range(num_events):
            if i < num_events - 1:
                df.loc[group.index[i], 'next_event_time'] = (timestamps[i + 1] - timestamps[i]).astype('timedelta64[s]').astype(np.int32)
            df.loc[group.index[i], 'remaining_time'] = (timestamps[-1] - timestamps[i]).astype('timedelta64[s]').astype(np.int32)

    return df


helpdesk_df, helpdesk_traces = load_helpdesk()

# Apply time feature calculation
helpdesk_df = calculate_time_features(helpdesk_df)

# One-hot encode the activity column
helpdesk_df_onehot = pd.get_dummies(helpdesk_df, columns=['Activity'])

# Initialize the scaler
scaler = MinMaxScaler()

# Normalize the next_event_time and remaining_time columns
helpdesk_df[['next_event_time', 'remaining_time']] = scaler.fit_transform(helpdesk_df[['next_event_time', 'remaining_time']])


def generate_labels(df):
    # Initialize labels
    df['next_activity'] = df['Activity'].shift(-1)

    # For the last event in each case, set next_activity as NaN or a special label
    for case_id, group in df.groupby('Case ID'):
        df.loc[group.index[-1], 'next_activity'] = 'END'  # Marking last event's next activity as 'END'

    return df


# Apply label generation
helpdesk_df = generate_labels(helpdesk_df)


# Assuming helpdesk_df contains the following columns:
# 'activity_<one-hot columns>', 'next_event_time', 'remaining_time', 'next_activity'
# Convert the columns to PyTorch tensors

# Features (activity sequences in one-hot encoded format)
X = torch.tensor(helpdesk_df_onehot.filter(like='Activity_').values, dtype=torch.float32)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the next_activity column
helpdesk_df['next_activity_encoded'] = label_encoder.fit_transform(helpdesk_df['next_activity'])

# Now you can convert the encoded labels into tensors
y_activity = torch.tensor(helpdesk_df['next_activity_encoded'].values, dtype=torch.long)

# Next event time (target for regression)
y_next_time = torch.tensor(helpdesk_df['next_event_time'].values, dtype=torch.float32)

# Remaining time (target for regression)
y_remaining_time = torch.tensor(helpdesk_df['remaining_time'].values, dtype=torch.float32)
