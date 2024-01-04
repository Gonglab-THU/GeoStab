import os
import pandas as pd
from biopandas.pdb import PandasPdb
from Bio.PDB import PDBParser, PDBIO

def process_mut_info(mut_info_csv_file, wt_data_folder, mut_data_folder):
    
    data = pd.read_csv(mut_info_csv_file, index_col = 0)
    assert data.columns.tolist() == ['metric', 'version', 'pdb_id', 'model_id', 'chain_id', 'pdb_seq', 'seq', 'mut_pos', 'mut_res', 'pH', 'temperature']
    assert data.loc['test', 'metric'] in ['fitness', 'ddGdTm']
    assert data.loc['test', 'version'] in ['3D', 'Seq']
    
    data[['mut_pos', 'pH', 'temperature']] = data[['mut_pos', 'pH', 'temperature']].astype(int)
    mut_pos = data.loc['test', 'mut_pos']
    mut_res = data.loc['test', 'mut_res']
    
    if data.loc['test', 'version'] == '3D':
        pdb_id = data.loc['test', 'pdb_id'].upper()
        model_id = data.loc['test', 'model_id']
        chain_id = data.loc['test', 'chain_id']
        PandasPdb().fetch_pdb(pdb_code = pdb_id).get_model(model_index = model_id).to_pdb(path = f'{wt_data_folder}/relaxed.pdb', records = ['ATOM'])
        io = PDBIO()
        io.set_structure(PDBParser().get_structure('', f'{wt_data_folder}/relaxed.pdb')[0][chain_id])
        io.save(f'{wt_data_folder}/relaxed.pdb')
        os.system(f'pdb_fixinsert {wt_data_folder}/relaxed.pdb | pdb_reres -0 | pdb_chain -A | pdb_tidy > {wt_data_folder}/tmp.pdb')
        os.system(f'mv {wt_data_folder}/tmp.pdb {wt_data_folder}/relaxed.pdb')
        
        # write fasta file
        wt_seq = data.loc['test', 'pdb_seq']
        wt_seq_list = list(wt_seq)
        mut_seq_list = wt_seq_list.copy()
        mut_seq_list[mut_pos] = mut_res
        mut_seq = ''.join(mut_seq_list)
        with open(f'{wt_data_folder}/result.fasta', 'w') as f:
            f.write(f'>result\n{wt_seq}')
        with open(f'{mut_data_folder}/result.fasta', 'w') as f:
            f.write(f'>result\n{mut_seq}')
        
        # write individual_list.txt file
        with open(f'{mut_data_folder}/individual_list.txt', 'w') as f:
            f.write('{}A{}{};'.format(wt_seq[mut_pos], mut_pos, mut_seq[mut_pos]))
    
    else:
        # write fasta file
        wt_seq = data.loc['test', 'seq']
        wt_seq_list = list(wt_seq)
        mut_seq_list = wt_seq_list.copy()
        mut_seq_list[mut_pos] = mut_res
        mut_seq = ''.join(mut_seq_list)
        with open(f'{wt_data_folder}/result.fasta', 'w') as f:
            f.write(f'>result\n{wt_seq}')
        with open(f'{mut_data_folder}/result.fasta', 'w') as f:
            f.write(f'>result\n{mut_seq}')
        
        # write individual_list.txt file
        with open(f'{mut_data_folder}/individual_list.txt', 'w') as f:
            f.write('{}A{}{};'.format(wt_seq[mut_pos], mut_pos, mut_seq[mut_pos]))

data = pd.DataFrame()
data.loc['test', 'metric'] = 'ddGdTm'
data.loc['test', 'version'] = '3D'
data.loc['test', 'pdb_id'] = '1CTS'
data.loc['test', 'model_id'] = '1'
data.loc['test', 'chain_id'] = 'A'
data.loc['test', 'pdb_seq'] = 'ASSTNLKDILADLIPKEQARIKTFRQQHGNTVVGQITVDMMYGGMRGMKGLVYETSVLDPDEGIRFRGYSIPECQKMLPKAKGGEEPLPEGLFWLLVTGQIPTEEQVSWLSKEWAKRAALPSHVVTMLDNFPTNLHPMSQLSAAITALNSESNFARAYAEGIHRTKYWELIYEDCMDLIAKLPCVAAKIYRNLYREGSSIGAIDSKLDWSHNFTNMLGYTDAQFTELMRLYLTIHSDHEGGNVSAHTSHLVGSALSDPYLSFAAAMNGLAGPLHGLANQEVLVWLTQLQKEVGKDVSDEKLRDYIWNTLNSGRVVPGYGHAVLRKTDPRYTCQREFALKHLPHDPMFKLVAQLYKIVPNVLLEQGKAKNPWPNVDAHSGVLLQYYGMTEMNYYTVLFGVSRALGVLAQLIWSRALGFPLERPKSMSTDGLIKLVDSK'
data.loc['test', 'seq'] = ''
data.loc['test', 'mut_pos'] = 273
data.loc['test', 'mut_res'] = 'R'
data.loc['test', 'pH'] = 7
data.loc['test', 'temperature'] = 25
os.system('mkdir -p ./example_3D/wt_data')
os.system('mkdir -p ./example_3D/mut_data')
data.to_csv('./example_3D/mut_info.csv')

process_mut_info('./example_3D/mut_info.csv', './example_3D/wt_data', './example_3D/mut_data')


data = pd.DataFrame()
data.loc['test', 'metric'] = 'ddGdTm'
data.loc['test', 'version'] = 'Seq'
data.loc['test', 'pdb_id'] = ''
data.loc['test', 'model_id'] = ''
data.loc['test', 'chain_id'] = ''
data.loc['test', 'pdb_seq'] = ''
data.loc['test', 'seq'] = 'ASSTNLKDILADLIPKEQARIKTFRQQHGNTVVGQITVDMMYGGMRGMKGLVYETSVLDPDEGIRFRGYSIPECQKMLPKAKGGEEPLPEGLFWLLVTGQIPTEEQVSWLSKEWAKRAALPSHVVTMLDNFPTNLHPMSQLSAAITALNSESNFARAYAEGIHRTKYWELIYEDCMDLIAKLPCVAAKIYRNLYREGSSIGAIDSKLDWSHNFTNMLGYTDAQFTELMRLYLTIHSDHEGGNVSAHTSHLVGSALSDPYLSFAAAMNGLAGPLHGLANQEVLVWLTQLQKEVGKDVSDEKLRDYIWNTLNSGRVVPGYGHAVLRKTDPRYTCQREFALKHLPHDPMFKLVAQLYKIVPNVLLEQGKAKNPWPNVDAHSGVLLQYYGMTEMNYYTVLFGVSRALGVLAQLIWSRALGFPLERPKSMSTDGLIKLVDSK'
data.loc['test', 'mut_pos'] = 273
data.loc['test', 'mut_res'] = 'R'
data.loc['test', 'pH'] = 7
data.loc['test', 'temperature'] = 25
os.system('mkdir -p ./example_Seq/wt_data')
os.system('mkdir -p ./example_Seq/mut_data')
data.to_csv('./example_Seq/mut_info.csv')

process_mut_info('./example_Seq/mut_info.csv', './example_Seq/wt_data', './example_Seq/mut_data')
