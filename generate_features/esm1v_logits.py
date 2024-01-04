import click
import torch
from Bio import SeqIO

@click.command()
@click.option('--model_index', required = True, type = str)
@click.option('--fasta_file', required = True, type = str)
@click.option('--saved_folder', required = True, type = str)
def main(model_index, fasta_file, saved_folder):
    seq = str(list(SeqIO.parse(fasta_file, 'fasta'))[0].seq)
    with torch.no_grad():
        model, alphabet = torch.hub.load('facebookresearch/esm:main', f'esm1v_t33_650M_UR90S_{model_index}')
        model = model.eval().to('cpu')
        
        batch_converter = alphabet.get_batch_converter()
        data = [('protein', seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        
        token_probs = torch.log_softmax(model(batch_tokens)['logits'], dim = -1)
        logits_33 = token_probs[0, 1:-1, :].detach().cpu().clone()
        
        # logits 33 dim -> 20 dim
        amino_acid_list = list('ARNDCQEGHILKMFPSTWYV')
        esm1v_amino_acid_dict = {}
        for i in amino_acid_list:
            esm1v_amino_acid_dict[i] = alphabet.get_idx(i)
        
        logits_20 = torch.zeros((logits_33.shape[0], 20))
        for wt_pos, wt_amino_acid in enumerate(seq):
            for mut_pos, mut_amino_acid in enumerate(amino_acid_list):
                logits_20[wt_pos, mut_pos] = logits_33[wt_pos, esm1v_amino_acid_dict[mut_amino_acid]] - logits_33[wt_pos, esm1v_amino_acid_dict[wt_amino_acid]]
        
        torch.save(logits_20.detach().cpu().clone(), f'{saved_folder}/esm1v-{model_index}.pt')

if __name__ == '__main__':
    main()
