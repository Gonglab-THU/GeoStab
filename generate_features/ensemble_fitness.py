import click
import torch
import pickle
import pandas as pd

@click.command()
@click.option('--af2_pickle_file', required = False, type = str)
@click.option('--saved_folder', required = True, type = str)
def main(af2_pickle_file, saved_folder):
    mut_info = pd.read_csv(f'{saved_folder}/mut_info.csv', index_col = 0)
    version = mut_info.loc['test', 'version']
    length = len(mut_info.loc['test', 'seq']) if version == 'Seq' else len(mut_info.loc['test', 'pdb_seq'])
    
    wt_folder = f'{saved_folder}/wt_data'
    data = {}
    
    wt_dynamic_embedding = torch.load(f'{wt_folder}/esm2.pt')
    wt_fixed_embedding = torch.load(f'{wt_folder}/fixed_embedding.pt')
    wt_pair = torch.load(f'{wt_folder}/pair.pt')
    wt_pos14 = torch.load(f'{wt_folder}/coordinate.pt')['pos14']
    wt_atom_mask = torch.load(f'{wt_folder}/coordinate.pt')['pos14_mask'].all(dim = -1)
    
    if version == '3D':
        plddt = torch.ones(length)
        wt_seq = mut_info.loc['test', 'pdb_seq']
    else:
        with open(af2_pickle_file, 'rb') as f:
            tmp = pickle.load(f)
        plddt = torch.from_numpy(tmp['plddt'] / 100).float()
        wt_seq = mut_info.loc['test', 'seq']
    
    # all logits
    all_logits = torch.cat([torch.load(f'{wt_folder}/esm1v-{i}.pt').unsqueeze(0) for i in range(1, 6)], dim = 0)
    
    data['fitness'] = {
        'dynamic_embedding': wt_dynamic_embedding.unsqueeze(0),
        'fixed_embedding': torch.cat((wt_fixed_embedding, plddt.unsqueeze(-1)), dim = -1).unsqueeze(0),
        'pair': wt_pair.unsqueeze(0),
        'pos14': wt_pos14.unsqueeze(0),
        'atom_mask': wt_atom_mask.unsqueeze(0),
        'logits': all_logits.unsqueeze(0),
        'seq': wt_seq
    }
    torch.save(data, f'{saved_folder}/ensemble.pt')

if __name__ == '__main__':
    main()
