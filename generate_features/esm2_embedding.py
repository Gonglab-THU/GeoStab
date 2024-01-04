import click
import torch
from Bio import SeqIO

@click.command()
@click.option('--fasta_file', required = True, type = str)
@click.option('--saved_folder', required = True, type = str)
def main(fasta_file, saved_folder):
    with torch.no_grad():
        model, alphabet = torch.hub.load('facebookresearch/esm:main', 'esm2_t33_650M_UR50D')
        model = model.eval().to('cpu')
        
        batch_converter = alphabet.get_batch_converter()
        data = [('protein', str(list(SeqIO.parse(fasta_file, 'fasta'))[0].seq))]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        
        result = model(batch_tokens.to('cpu'), repr_layers = [33], return_contacts = False)
        representations = result['representations'][33][0, 1:-1, :]
        torch.save(representations.detach().cpu().clone(), f'{saved_folder}/esm2.pt')

if __name__ == '__main__':
    main()
