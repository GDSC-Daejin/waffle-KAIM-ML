import torch

def train_model(model, X_train, y_train, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

def evaluate_model(model, X_test, y_test, criterion):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        mse = criterion(predictions, y_test)
        rmse = torch.sqrt(mse)
    return rmse.item()
