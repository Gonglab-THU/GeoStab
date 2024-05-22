import os
import torch
import numpy as np
import pandas as pd
from model import PretrainModel
from dms import ProcessingData, train_model, validation_model, test_model

#######################################################################
# predifined parameters
#######################################################################

node_dims = [32, 64, 128]
num_layers = [1, 2]
n_heads = [4, 8]
pair_dims = [32, 64]

device = 0
seed = 0
early_stop = 20

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

#######################################################################
# data
#######################################################################

train_pt = torch.load('data_af2.pt')
train_csv = pd.DataFrame(index = train_pt.keys())

# generate random subsample index
for name in train_pt.keys():
    label = train_pt[name]['label']
    
    # get all no-nan index
    no_nan_x, no_nan_y = torch.where(~torch.isnan(label))
    length = len(no_nan_x)
    
    np.random.seed(seed)
    no_nan_index_shuffle = np.random.permutation(length)
    
    train_valid_bound = int(length * 0.7)
    valid_test_bound = int(length * 0.8)
    
    # train_index, validation_index, test_index
    train_csv.loc[name, 'train'] = str([no_nan_x[no_nan_index_shuffle[:train_valid_bound]].tolist(), no_nan_y[no_nan_index_shuffle[:train_valid_bound]].tolist()])
    train_csv.loc[name, 'validation'] = str([no_nan_x[no_nan_index_shuffle[train_valid_bound:valid_test_bound]].tolist(), no_nan_y[no_nan_index_shuffle[train_valid_bound:valid_test_bound]].tolist()])
    train_csv.loc[name, 'test'] = str([no_nan_x[no_nan_index_shuffle[valid_test_bound:]].tolist(), no_nan_y[no_nan_index_shuffle[valid_test_bound:]].tolist()])

test_dataset = ProcessingData(train_csv, train_pt, 'test')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False)

train_dataset = ProcessingData(train_csv, train_pt, 'train')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 1, shuffle = True)

validation_dataset = ProcessingData(train_csv, train_pt, 'validation')
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = 1, shuffle = False)

#######################################################################
# train
#######################################################################

for node_dim in node_dims:
    for num_layer in num_layers:
        for n_head in n_heads:
            for pair_dim in pair_dims:
                file = f'all_model/node_dim_{node_dim}-num_layer_{num_layer}-n_head_{n_head}-pair_dim_{pair_dim}'
                os.system(f'mkdir -p {file}')
                
                model = PretrainModel(node_dim, n_head, pair_dim, num_layer).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, factor = 0.5, patience = 5, verbose = True)
                
                best_loss = float('inf')
                stop_step = 0
                loss = pd.DataFrame()
                for epoch in range(200):
                    
                    train_loss = train_model(model, optimizer, train_loader)
                    validation_loss = validation_model(model, validation_loader)
                    
                    loss.loc[epoch, 'train_loss'] = train_loss
                    loss.loc[epoch, 'validation_loss'] = validation_loss
                    
                    print(loss)
                    
                    scheduler.step(validation_loss)
                    if validation_loss < best_loss:
                        stop_step = 0
                        best_loss = validation_loss
                        torch.save(model, f'{file}/best.pt')
                    else:
                        stop_step += 1
                        if stop_step >= early_stop:
                            break
                    loss.to_csv(f'{file}/loss.csv')
                
                test_csv = pd.DataFrame(index = train_pt.keys())
                model = torch.load(f'{file}/best.pt', map_location = lambda storage, loc: storage.cuda(device))
                model.eval()
                _, test_corr_dict = test_model(model, test_loader)
                test_csv['pred'] = test_csv.index.map(test_corr_dict)
                test_csv.to_csv(f'{file}/pred.csv')
