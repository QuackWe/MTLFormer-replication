from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from data_HelpDesk import X, y_activity, y_next_time, y_remaining_time


# Set batch size
batch_size = 64


class EventLogDataset(Dataset):
    def __init__(self, X, y_activity, y_next_time, y_remaining_time):
        self.X = X
        self.y_activity = y_activity
        self.y_next_time = y_next_time
        self.y_remaining_time = y_remaining_time

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'sequence': self.X[idx],
            'next_activity': self.y_activity[idx],
            'next_event_time': self.y_next_time[idx],
            'remaining_time': self.y_remaining_time[idx]
        }


# Split data into training and validation sets (80% train, 20% validation)
X_train, X_val, y_activity_train, y_activity_val, y_next_time_train, y_next_time_val, y_remaining_time_train, y_remaining_time_val = train_test_split(
    X, y_activity, y_next_time, y_remaining_time, test_size=0.2, random_state=42)


# Create datasets
train_dataset = EventLogDataset(X_train, y_activity_train, y_next_time_train, y_remaining_time_train)
val_dataset = EventLogDataset(X_val, y_activity_val, y_next_time_val, y_remaining_time_val)


# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
