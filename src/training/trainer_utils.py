## Creating the traiing and testing function automatically 
import torch 
import torch.nn as nn 
from src.utils.metrics import deep_evaluation
import os 
import numpy as np 

def model_train_and_validate(model,epoches, train_loader, validation_loader, optimizer, loss_method, device, saving_directory, model_name):
    model.to(device)
    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": []} 
    for epoch in range(epoches):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            samples, features = batch 
            samples = samples.to(device)
            features = features.to(device)
            output = model(samples)
            loss = loss_method(output, features)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        training_loss = train_loss/len(train_loader)
        history['train_loss'].append(training_loss)
        print(f"Epoch {epoch+1}/{epoches}, Training Loss {training_loss}")
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for batch in validation_loader:
                samples, features = batch 
                samples = samples.to(device)
                features = features.to(device)
                output = model(samples)
                loss = loss_method(output, features)
                validation_loss += loss.item()
        validation_loss = validation_loss/len(validation_loader)
        history['val_loss'].append(validation_loss)
        print(f"Epoch {epoch+1}/{epoches}, Validation Loss {validation_loss}")
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            torch.save({
                "epoch": epoch+1, 
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": training_loss,
                "val_loss": validation_loss, 
                "history": history

            }, f"{saving_directory}{model_name}_best.pt")
    return history


def model_test(model, test_loader, loss_method, device):
    model.to(device)
    test_loss = 0.0
    true_values = []
    predicted = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            samples, features = batch 
            samples = samples.to(device)
            features = features.to(device)
            output = model(samples)
            predicted.append(output)
            loss = loss_method(output, features)
            test_loss += loss.item()
            true_values.append(features)
        testing_loss = test_loss/len(test_loader)
        mae, rmse, r2 = deep_evaluation(predicted, true_values)
        predicted_flat = torch.cat(predicted).detach().cpu().numpy().flatten().tolist()
        true_values_flat = torch.cat(true_values).detach().cpu().numpy().flatten().tolist()
        return testing_loss, true_values_flat, predicted_flat, mae, rmse, r2
    

