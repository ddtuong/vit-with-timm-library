from configuration import *

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    losses = 0.0
    accuracy_scores = 0.0
    precision_scores = 0.0
    recall_scores = 0.0

    for (data, target) in tqdm(train_loader, total=len(train_loader)):
        target = target.unsqueeze(1).float()
        if device == "cuda":
            data, target = data.cuda(), target.cuda()
        output = model(data)
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        losses += loss.item()
        output = torch.sigmoid(output)
        output = (output > 0.5).detach().cpu().numpy().astype(int)
        target = target.detach().cpu().numpy().astype(int)
        accuracy_scores += metrics.accuracy_score(target, output)
        precision_scores += metrics.precision_score(target, output, average='binary', zero_division=0)
        recall_scores += metrics.recall_score(target, output, pos_label=1, average='binary', zero_division=0)

    return losses/len(train_loader), accuracy_scores/len(train_loader), precision_scores/len(train_loader), recall_scores/len(train_loader) 

def validate_one_epoch(model, valid_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        losses = 0.0
        accuracy_scores = 0.0
        precision_scores = 0.0
        recall_scores = 0.0
    
        for (data, target) in tqdm(valid_loader, total=len(valid_loader)):
            target = target.unsqueeze(1).float()
            if device == "cuda":
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
    
            losses += loss.item()
            output = torch.sigmoid(output)
            output = (output > 0.5).detach().cpu().numpy().astype(int)
            target = target.detach().cpu().numpy().astype(int)
            accuracy_scores += metrics.accuracy_score(target, output)
            precision_scores += metrics.precision_score(target, output, average='binary', zero_division=0)
            recall_scores += metrics.recall_score(target, output, average='binary', zero_division=0)
    
        return losses/len(valid_loader), accuracy_scores/len(valid_loader), precision_scores/len(valid_loader), recall_scores/len(valid_loader) 
             

def fit(model, train_loader, valid_loader, epochs, criterion, optimizer, device):
    train_losses = []
    train_accs = []
    train_pres = []
    train_recs = []
    
    valid_losses = []
    valid_accs = []
    valid_pres = []
    valid_recs = []
    best_loss = 1000
    
    for epoch in range(1, epochs+1):
        gc.collect()
        loss, acc, pre, rec = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc, valid_pre, valid_rec = validate_one_epoch(model, valid_loader, criterion, device)
        print(f"Training loss: {loss}, Accuracy: {acc}, Precision: {pre}, Recall: {rec}")
        print(f"Validation loss: {valid_loss}, Validation Accuracy: {valid_acc}, Validation Precision: {valid_pre}, Validation Recall: {valid_rec}")
        
        # save best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            
        train_losses.append(loss)
        train_accs.append(acc)
        train_pres.append(pre)
        train_recs.append(rec)

        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        valid_pres.append(valid_pre)
        valid_recs.append(valid_rec)

    return {
        "train_loss": train_losses,
        "train_accuracy": train_accs,
        "train_precision": train_pres,
        "train_recall": train_recs,
        "valid_loss": valid_losses,
        "valid_accuracy": valid_accs,
        "valid_precision": valid_pres,
        "valid_recall": valid_recs
    }
        