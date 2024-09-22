import torch


# Evaluation for classification and regression tasks
def evaluate_model(model, dataloader):
    model.eval()
    total_activity_correct = 0
    total_samples = 0
    total_time_mae = 0
    total_remaining_mae = 0

    with torch.no_grad():
        for batch in dataloader:
            traces, activity_labels, time_labels, remaining_labels = batch

            # Forward pass
            activity_pred, time_pred, remaining_pred = model(traces)

            # Compute accuracy for next activity prediction (classification)
            _, predicted_activity = torch.max(activity_pred, 1)
            total_activity_correct += (predicted_activity == activity_labels).sum().item()

            # Compute MAE for next event time prediction
            total_time_mae += torch.abs(time_pred - time_labels).sum().item()

            # Compute MAE for remaining time prediction
            total_remaining_mae += torch.abs(remaining_pred - remaining_labels).sum().item()

            total_samples += activity_labels.size(0)

    # Accuracy and MAE scores
    accuracy = total_activity_correct / total_samples
    avg_time_mae = total_time_mae / total_samples
    avg_remaining_mae = total_remaining_mae / total_samples

    print(f'Accuracy for next activity: {accuracy:.4f}')
    print(f'MAE for next event time: {avg_time_mae:.4f}')
    print(f'MAE for remaining time: {avg_remaining_mae:.4f}')
