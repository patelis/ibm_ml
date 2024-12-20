import torch
from torch import nn
from tqdm.auto import tqdm

# Function to run a training step of the model and return loss and accuracy
def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer, 
               scheduler: torch.optim.lr_scheduler._LRScheduler | None,
               augmentations = False,
               device = "cpu"):
    
    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        if bool(augmentations):
            X = augmentations(X)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        if bool(scheduler):
            scheduler.step()    
        
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim = 1)
        train_acc += ((y_pred_class == y).sum().item() / len(y_pred))

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc

#Function to evaluate the model on the validation/test set and return loss and accuracy
def test_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               #optimizer: torch.optim.Optimizer, 
               device = "cpu"):
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_acc

# Save intermediate epochs of the model so that we can revert to the best checkpoint
def save_intermediate_epochs(model: torch.nn.Module, experiment_name: str, warm_up_epochs: int, epochs: int, save_dir: str = "intermediate_weights/"):
    import os
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f"{experiment_name}_warmup_{warm_up_epochs}_epochs_{epochs}.pt"))
    
    

# Function to train the model for several epochs
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          loss_fn: torch.nn.Module, 
          optimizer: torch.optim.Optimizer, 
          scheduler: bool,
          augmentations = False, 
          warm_up_epochs:int = 0,
          epochs:int = 5,
          save_intermediate_weights = False,
          save_intermediate_weights_loc = "intermediate_weights/",
          device = "cpu"): 
    results = {"train_loss": [], 
               "train_acc": [], 
               "test_loss": [], 
               "test_acc": []
               }

    if warm_up_epochs > 0:
        for param in model.parameters():
            param.requires_grad = False
        
        # Figure out how to parameterize the name of the last layer, currently updated the function to run EfficientNet
        # Maybe use list(model.children)??
        #for param in model.fc.parameters():
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 0.01, steps_per_epoch=len(train_dataloader),epochs=warm_up_epochs)

        for epoch in tqdm(range(warm_up_epochs)):
            train_loss, train_acc = train_step(model = model, 
                                           dataloader = train_dataloader,
                                           loss_fn = loss_fn, 
                                           optimizer = optimizer, 
                                           scheduler = scheduler, 
                                           augmentations = augmentations,
                                           device=device)
        
            test_loss, test_acc = test_step(model = model, 
                                               dataloader = test_dataloader,
                                               loss_fn = loss_fn, 
                                               device=device)

            print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {100*train_acc:.2f}% | Test loss: {test_loss:.4f} | Test acc: {100*test_acc:.2f}%")

            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
            
            if save_intermediate_weights:
                import os
                os.makedirs(save_intermediate_weights_loc, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_intermediate_weights_loc, f"{model.__class__.__name__}_warmup_{epoch}.pt"))
            
        for param in model.parameters():
            param.requires_grad = True

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 0.01, steps_per_epoch=len(train_dataloader),epochs=epochs)
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model = model, 
                                           dataloader = train_dataloader,
                                           loss_fn = loss_fn, 
                                           optimizer = optimizer, 
                                           scheduler = scheduler, 
                                           augmentations = augmentations,
                                           device=device)
        
        test_loss, test_acc = test_step(model = model, 
                                        dataloader = test_dataloader,
                                        loss_fn = loss_fn, 
                                        device=device)

        print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {100*train_acc:.2f}% | Test loss: {test_loss:.4f} | Test acc: {100*test_acc:.2f}%")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
        if save_intermediate_weights:
            import os
            os.makedirs(save_intermediate_weights_loc, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_intermediate_weights_loc, f"{model.__class__.__name__}_epoch_{epoch}.pt"))

    return results


# Return test accuracy on unseen data

def test_accuracy(model: torch.nn.Module, 
                  dataloader: torch.utils.data.DataLoader, 
                  device = "cpu"):
    
    model.eval()
    test_acc = 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    test_acc = test_acc / len(dataloader)

    return test_acc