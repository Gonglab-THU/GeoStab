import torch
from metrics import spearman_corr
from loss import spearman_loss

class ProcessingData(torch.utils.data.Dataset):
    
    def __init__(self, csv, pt, dataset_type):
        
        self.csv = csv
        self.pt = pt
        self.dataset_type = dataset_type
    
    def __len__(self):
        
        return len(self.csv)
    
    def __getitem__(self, index):
        
        name = self.csv.iloc[index].name
        data = self.pt[name]
        label = data['label']
        select_index = self.csv.loc[name, self.dataset_type]
        
        return data, label, select_index, name

def to_gpu(obj, device):
    if isinstance(obj, torch.Tensor):
        try:
            return obj.to(device = device, non_blocking = True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [to_gpu(i, device = device) for i in obj]
    elif isinstance(obj, tuple):
        return (to_gpu(i, device = device) for i in obj)
    elif isinstance(obj, dict):
        return {i: to_gpu(j, device = device) for i, j in obj.items()}
    else:
        return obj

def train_model(model, optimizer, loader):
    
    model.train()
    device = next(model.parameters()).device
    epoch_loss = 0
    for data, label, select_index, name in loader:
        data, label = to_gpu(data, device), to_gpu(label, device)
        
        optimizer.zero_grad()
        pred = model(data)
        if '_wt' in name[0]:
            loss = spearman_loss(pred[0][eval(select_index[0])].unsqueeze(0), label[0][eval(select_index[0])].unsqueeze(0), 1e-1, 'kl')
        else:
            loss = spearman_loss(pred[0][eval(select_index[0])].unsqueeze(0), label[0][eval(select_index[0])].unsqueeze(0), 1e-2, 'kl')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), norm_type = 2, max_norm = 10, error_if_nonfinite = True)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def validation_model(model, loader):
    
    model.eval()
    device = next(model.parameters()).device
    epoch_loss = 0
    with torch.no_grad():
        for data, label, select_index, _ in loader:
            data, label = to_gpu(data, device), to_gpu(label, device)
            
            pred = model(data)
            loss = -spearman_corr(pred[0][eval(select_index[0])], label[0][eval(select_index[0])])
            epoch_loss += loss.item()
    return epoch_loss / len(loader)

def test_model(model, loader):
    
    model.eval()
    device = next(model.parameters()).device
    corr_sum = 0
    corr_dict = {}
    with torch.no_grad():
        for data, label, select_index, name in loader:
            data, label = to_gpu(data, device), to_gpu(label, device)
            
            pred = model(data)
            corr = spearman_corr(pred[0][eval(select_index[0])], label[0][eval(select_index[0])]).item()
            corr_sum += corr
            corr_dict[name[0]] = corr
    return corr_sum / len(loader), corr_dict
