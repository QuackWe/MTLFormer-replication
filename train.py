from loss import multitask_loss


# Training loop
def train_model(model, dataloader, optimizer, weights, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            # Get data for each task
            traces, activity_labels, time_labels, remaining_labels = batch

            # Forward pass
            activity_pred, time_pred, remaining_pred = model(traces)

            # Compute loss
            loss = multitask_loss(activity_pred, time_pred, remaining_pred,
                                  activity_labels, time_labels, remaining_labels, weights)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

