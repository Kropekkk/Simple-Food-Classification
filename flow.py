import torch
from typing import Dict, List

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,) -> Dict[str, List]:

  results = {"train_loss": [],
      "train_acc": [],-
      "test_loss": [],
      "test_acc": []
  }

  train_loss, train_acc = 0, 0
  test_loss, test_acc = 0, 0
  
  for epoch in range(epochs):
    model.train()

    for batch, (X, y) in enumerate(train_dataloader):
      y_pred = model(X)

      loss = loss_fn(y_pred, y)
      train_loss += loss.item() 

      optimizer.zero_grad()

      loss.backward()

      optimizer.step()

      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    model.eval()
    with torch.inference_mode():
      for batch, (X, y) in enumerate(test_dataloader):
          test_pred_logits = model(X)

          loss = loss_fn(test_pred_logits, y)
          test_loss += loss.item()
          
          test_pred_labels = test_pred_logits.argmax(dim=1)
          test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    #Adjust accuracy and loss to lenght of batches
    train_loss = train_loss / len(train_dataloader)
    train_acc = train_acc / len(train_dataloader)
    test_loss = test_loss / len(test_dataloader)
    test_acc = test_acc / len(test_dataloader)

    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

  return results