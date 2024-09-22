from torch import nn


# Loss functions
def multitask_loss(activity_pred, time_pred, remaining_pred, activity_label, time_label, remaining_label, weights):
    # Task A: Cross Entropy Loss for Next Activity Prediction (classification)
    activity_loss = nn.CrossEntropyLoss()(activity_pred, activity_label)

    # Task B: Mean Absolute Error (MAE) for Next Event Time Prediction (regression)
    time_loss = nn.L1Loss()(time_pred, time_label)

    # Task C: MAE for Remaining Time Prediction (regression)
    remaining_loss = nn.L1Loss()(remaining_pred, remaining_label)

    # Total loss = Weighted sum of losses
    total_loss = weights[0] * activity_loss + weights[1] * time_loss + weights[2] * remaining_loss
    return total_loss
