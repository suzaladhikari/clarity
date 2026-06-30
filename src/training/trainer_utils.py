## Creating the traiing and testing function automatically 
import torch 
import torch.nn as nn 

def model_train_and_validate(model,epoches, train_loader, validation_loader, optimizer, loss_method, device):
    model.to(device)
    for epoch in range(epoches):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch.to(device)
            optimizer.zero_grad()
            samples, features = batch 
            samples.to(device)
            features.to(device)
            output = model(samples)
            loss = loss_method(output, features)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        training_loss = train_loss/len(train_loader.dataset)

        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for batch in validation_loader:
                batch.to(device)
                samples, features = batch 
                samples.to(device)
                features.to(device)
                output = model(samples)
                loss = loss_method(output, features)
                validation_loss += loss.item()
            validatoion_loss = validation_loss/len(validation_loader.dataset)
    return training_loss, validatoion_loss

