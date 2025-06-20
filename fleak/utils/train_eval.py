import torch




ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

def train(model, device, train_loader, optimizer, criterion,sigma):
    torch.autograd.set_detect_anomaly(True)

    model.train()
    # batch training
    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)


        # clear tensor grad
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.7)

        optimizer.step()




def evaluate(model, device, eval_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for features, labels in eval_loader:
            features, labels = features.to(device), labels.to(device)
            output = model(features)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
    return correct

def ternarize_gradient(grad):
    s = grad.abs().mean()
    return torch.sign(grad) * s



