import os
import click
import torch
import pandas as pd


@click.command()
@click.option("--ensemble_feature", required=True, type=str)
@click.option("--saved_folder", required=True, type=str)
def main(ensemble_feature, saved_folder):
    path = os.path.split(os.path.realpath(__file__))[0]
    model = torch.load(f"{path}/model.pt", map_location=torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        features = torch.load(ensemble_feature)["fitness"]
        data = pd.DataFrame(model(features)[0].T.numpy())
        data.index = list("ARNDCQEGHILKMFPSTWYV")
        data.columns = list(features["seq"])
        data.to_csv(f"{saved_folder}/result_fitness.csv")


if __name__ == "__main__":
    main()
