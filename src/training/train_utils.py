import torch
from tqdm import tqdm

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    total_loss = 0

    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader):

        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.sigmoid(outputs).detach().cpu().numpy().reshape(-1)
        labels_np = labels.detach().cpu().numpy().reshape(-1)

        all_preds.extend(preds)
        all_labels.extend(labels_np)

    return total_loss / len(loader), all_labels, all_preds


def validate(model, loader, criterion, device):

    model.eval()

    total_loss = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = torch.sigmoid(outputs).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), all_labels, all_preds
