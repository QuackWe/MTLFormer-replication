import pandas as pd
import numpy as np
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from sklearn.preprocessing import MinMaxScaler


def load_bpic():
    # Load BPIC2012 XES
    bpic2012_log = xes_importer.apply('data/BPIC2012.xes')

    # Convert the log to a dataframe
    bpic2012_df = pm4py.convert_to_dataframe(bpic2012_log)

    # Sorting and grouping by case
    bpic2012_df = bpic2012_df.sort_values(by=['case:concept:name', 'time:timestamp'])

    bpic2012_traces = bpic2012_df.groupby('case:concept:name').agg(list)

    # Display one trace
    return bpic2012_df, bpic2012_traces

def calculate_time_features(df):
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])  # Convert to datetime

    # Initialize time columns
    df['next_event_time'] = 0
    df['remaining_time'] = 0

    # Iterate through each case to calculate next event time and remaining time
    for case_id, group in df.groupby('case:concept:name'):
        timestamps = group['time:timestamp'].values
        num_events = len(timestamps)

        # Calculate next event time and remaining time for each event in the trace
        for i in range(num_events):
            if i < num_events - 1:
                df.loc[group.index[i], 'next_event_time'] = (timestamps[i + 1] - timestamps[i]).astype('timedelta64[s]').astype(np.int32)
            df.loc[group.index[i], 'remaining_time'] = (timestamps[-1] - timestamps[i]).astype('timedelta64[s]').astype(np.int32)

    return df


bpic2012_df, bpic2012_traces = load_bpic()

# Apply time feature calculation
bpic2012_df = calculate_time_features(bpic2012_df)

# One-hot encode the activity column
bpic2012_df_onehot = pd.get_dummies(bpic2012_df, columns=['concept:name'])

# Initialize the scaler
scaler = MinMaxScaler()

# Normalize the next_event_time and remaining_time columns
bpic2012_df[['next_event_time', 'remaining_time']] = scaler.fit_transform(bpic2012_df[['next_event_time', 'remaining_time']])


def generate_labels(df):
    # Initialize labels
    df['next_activity'] = df['concept:name'].shift(-1)

    # For the last event in each case, set next_activity as NaN or a special label
    for case_id, group in df.groupby('case:concept:name'):
        df.loc[group.index[-1], 'next_activity'] = 'END'  # Marking last event's next activity as 'END'

    return df


# Apply label generation
bpic2012_df = generate_labels(bpic2012_df)