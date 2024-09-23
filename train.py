from loss import multitask_loss
import torch


# Training loop
def train_model(model, dataloader, optimizer, weights, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            # Get data for each task
            sequences = batch['sequence']  # Activity sequences (input features)
            next_activity_labels = batch['next_activity']  # Next activity (classification labels)
            next_event_time_labels = batch['next_event_time']  # Next event time (regression labels)
            remaining_time_labels = batch['remaining_time']  # Remaining time (regression labels)

            # Forward pass
            activity_pred, time_pred, remaining_pred = model(sequences)

            # Squeeze the prediction to remove the extra dimension
            time_pred = torch.squeeze(time_pred)  # For next_event_time prediction
            remaining_pred = torch.squeeze(remaining_pred)  # For remaining_time prediction

            # Compute loss
            loss = multitask_loss(activity_pred, time_pred, remaining_pred,
                                  next_activity_labels, next_event_time_labels, remaining_time_labels, weights)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")
    return epoch, num_epochs, total_loss
