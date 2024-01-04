import os
import click
import torch

@click.command()
@click.option('--ensemble_feature', required = True, type = str)
@click.option('--saved_folder', required = True, type = str)
def main(ensemble_feature, saved_folder):
    path = os.path.split(os.path.realpath(__file__))[0]
    model = torch.load(f'{path}/model.pt', map_location = torch.device('cpu'))
    model.eval()
    data = torch.load(ensemble_feature)['ddG']
    with torch.no_grad():
        result = model(data['wt'], data['mut']).item()
        with open(f'{saved_folder}/result_ddG.txt', 'w') as f:
            f.write(f'{result}')

if __name__ == '__main__':
    main()
