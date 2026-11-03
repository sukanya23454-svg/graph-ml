import torch


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    P, Y = [], []
    for b in loader:
        P.append(model(b.to(dev)).cpu())
        Y.append(b.y.cpu())
    return torch.cat(P).numpy(), torch.cat(Y).numpy()
