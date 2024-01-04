import click
import torch
import pickle
import pandas as pd

num_neighbor = 32

ATOM_N, ATOM_CA, ATOM_C, ATOM_O, ATOM_CB = 0, 1, 2, 3, 4

amino_acid_list = list('ARNDCQEGHILKMFPSTWYV')
amino_acid_dict = {}
for index, value in enumerate(amino_acid_list):
    amino_acid_dict[value] = index

def mask_and_unsqueeze_dict(data, mask):
    out = {}
    for k, v in data.items():
        if k in ('fixed_embedding', 'dynamic_embedding', 'pos14', 'atom_mask', 'mut_pos'):
            out[k] = v[mask].unsqueeze(0)
        elif k in ('pair'):
            out[k] = v[mask][:, mask].unsqueeze(0)
        else:
            raise ValueError(f'Unknown key: {k}')
    return out

@click.command()
@click.option('--af2_pickle_file', required = False, type = str)
@click.option('--saved_folder', required = True, type = str)
def main(af2_pickle_file, saved_folder):
    mut_info = pd.read_csv(f'{saved_folder}/mut_info.csv', index_col = 0)
    version = mut_info.loc['test', 'version']
    length = len(mut_info.loc['test', 'seq']) if version == 'Seq' else len(mut_info.loc['test', 'pdb_seq'])
    mut_pos = int(mut_info.loc['test', 'mut_pos'])
    mut_res = mut_info.loc['test', 'mut_res']
    
    wt_folder = f'{saved_folder}/wt_data'
    mut_folder = f'{saved_folder}/mut_data'
    data = {}
    
    wt_dynamic_embedding = torch.load(f'{wt_folder}/esm2.pt')
    mut_dynamic_embedding = torch.load(f'{mut_folder}/esm2.pt')
    wt_fixed_embedding = torch.load(f'{wt_folder}/fixed_embedding.pt')
    mut_fixed_embedding = torch.load(f'{mut_folder}/fixed_embedding.pt')
    wt_pair = torch.load(f'{wt_folder}/pair.pt')
    mut_pair = torch.load(f'{mut_folder}/pair.pt')
    wt_pos14 = torch.load(f'{wt_folder}/coordinate.pt')['pos14']
    mut_pos14 = torch.load(f'{mut_folder}/coordinate.pt')['pos14']
    wt_atom_mask = torch.load(f'{wt_folder}/coordinate.pt')['pos14_mask'].all(dim = -1)
    mut_atom_mask = torch.load(f'{mut_folder}/coordinate.pt')['pos14_mask'].all(dim = -1)
    pH = torch.tensor(mut_info.loc['test', 'pH']).float()
    pH = torch.clamp(pH, 0, 11)
    temperature = torch.tensor(mut_info.loc['test', 'temperature']).float() / 10
    temperature = torch.clamp(temperature, 0, 12)
    
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
    
    # mutlogits
    mut_seq_list = list(wt_seq)
    mut_seq_list[mut_pos] = mut_res
    mut_seq = ''.join(mut_seq_list)
    mutlogits = torch.tensor([all_logits[i, mut_pos, amino_acid_dict[mut_res]].item() for i in range(5)])
    
    # logits + pH + plddt
    wt_lp_plddt = torch.cat((torch.tensor([pH])[None, :].repeat(length, 1), plddt.unsqueeze(-1)), dim = -1)
    mut_lp_plddt = torch.cat((torch.tensor([pH])[None, :].repeat(length, 1), plddt.unsqueeze(-1)), dim = -1)
    
    # logits + pH + temperature + plddt
    wt_lpt_plddt = torch.cat((torch.tensor([pH, temperature])[None, :].repeat(length, 1), plddt.unsqueeze(-1)), dim = -1)
    mut_lpt_plddt = torch.cat((torch.tensor([pH, temperature])[None, :].repeat(length, 1), plddt.unsqueeze(-1)), dim = -1)
    
    # mut_pos_list
    mut_pos_list = torch.LongTensor([amino_acid_dict[i] for i in wt_seq]) != torch.LongTensor([amino_acid_dict[i] for i in mut_seq])
    
    data['fitness'] = {
        'dynamic_embedding': wt_dynamic_embedding.unsqueeze(0),
        'fixed_embedding': torch.cat((wt_fixed_embedding, plddt.unsqueeze(-1)), dim = -1).unsqueeze(0),
        'pair': wt_pair.unsqueeze(0),
        'pos14': wt_pos14.unsqueeze(0),
        'atom_mask': wt_atom_mask.unsqueeze(0),
        'logits': all_logits.unsqueeze(0),
        'seq': wt_seq
    }
    data['ddG'] = {
        'wt': {
            'dynamic_embedding': wt_dynamic_embedding,
            'fixed_embedding': torch.cat((wt_fixed_embedding, wt_lpt_plddt), dim = -1),
            'pair': wt_pair,
            'pos14': wt_pos14,
            'atom_mask': wt_atom_mask,
            'mut_pos': mut_pos_list
        },
        'mut': {
            'dynamic_embedding': mut_dynamic_embedding,
            'fixed_embedding': torch.cat((mut_fixed_embedding, mut_lpt_plddt), dim = -1),
            'pair': mut_pair,
            'pos14': mut_pos14,
            'atom_mask': mut_atom_mask,
            'mut_pos': mut_pos_list
        }
    }
    data['dTm'] = {
        'wt': {
            'dynamic_embedding': wt_dynamic_embedding,
            'fixed_embedding': torch.cat((wt_fixed_embedding, wt_lp_plddt), dim = -1),
            'pair': wt_pair,
            'pos14': wt_pos14,
            'atom_mask': wt_atom_mask,
            'mut_pos': mut_pos_list
        },
        'mut': {
            'dynamic_embedding': mut_dynamic_embedding,
            'fixed_embedding': torch.cat((mut_fixed_embedding, mut_lp_plddt), dim = -1),
            'pair': mut_pair,
            'pos14': mut_pos14,
            'atom_mask': mut_atom_mask,
            'mut_pos': mut_pos_list
        }
    }
    
    # dist limit
    coor_CA = wt_pos14[:, ATOM_CA, :]
    mut_pos_coor_CA = coor_CA[mut_pos_list]
    diff = mut_pos_coor_CA.view(1, -1, 3) - coor_CA.view(-1, 1, 3)
    dist = torch.linalg.norm(diff, dim = -1)
    mask = torch.zeros([dist.shape[0]], dtype = torch.bool)
    mask[ dist.min(dim = 1)[0].argsort()[:num_neighbor] ] = True
    
    data['ddG']['wt'] = mask_and_unsqueeze_dict(data['ddG']['wt'], mask)
    data['ddG']['mut'] = mask_and_unsqueeze_dict(data['ddG']['mut'], mask)
    data['dTm']['wt'] = mask_and_unsqueeze_dict(data['dTm']['wt'], mask)
    data['dTm']['mut'] = mask_and_unsqueeze_dict(data['dTm']['mut'], mask)
    torch.save(data, f'{saved_folder}/ensemble.pt')

if __name__ == '__main__':
    main()
