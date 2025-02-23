import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, X_train, Y_train, epochs=100, batch_size=32):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_Y = Y_train[i:i+batch_size]
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model
