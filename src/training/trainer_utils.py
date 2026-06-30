## Creating the traiing and testing function automatically 
import torch 
import torch.nn as nn 
from src.utils.metrics import deep_evaluation

def model_train_and_validate(model,epoches, train_loader, validation_loader, optimizer, loss_method, device):
    model.to(device)
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
    return training_loss, validation_loss


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
        testing_loss = test_loss/len(test_loader.dataset)
        mae, rmse, r2 = deep_evaluation(predicted, true_values)
        return testing_loss, true_values, predicted, mae, rmse, r2
    