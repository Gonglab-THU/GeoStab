<h1 align="center">GeoStab</h1>
<p align="center">Improving the prediction of protein stability changes upon mutations by geometric learning and a pre-training strategy</p>

## Data


|   Dataset/Database   |     Type    |                          Composition                          |                               url                               |   Time  |
| :------------------: | :---------: | :-----------------------------------------------------------: | :-------------------------------------------------------------: | :-----: |
| DeepSequence dataset |   Fitness   |              34 single-point mutation DMS studies             |                                                                 | 2018-09 |
|        MaveDB        |   Fitness   |   The database contains fitness data of different mutations   |                    <https://www.mavedb.org/>                    | 2022-09 |
|  GeoFitness dataset  |   Fitness   |                        74 DMS datasets                        |                   [dms.csv](data/dms/dms.csv)                   | 2023-05 |
|      ProThermDB      | ΔΔG and ΔTm | The database contains ΔΔG and ΔTm data of different mutations |     <https://web.iitm.ac.in/bioinfo2/prothermdb/index.html>     | 2021-07 |
|      ThermoMutDB     | ΔΔG and ΔTm | The database contains ΔΔG and ΔTm data of different mutations |           <https://biosig.lab.uq.edu.au/thermomutdb/>           | 2021-07 |
|         S2648        |     ΔΔG     |          2648 single-point mutation from 131 proteins         | This file is included in [S8754.csv](data/ddG/S8754.csv) | 2009-08 |
|         S8754        |     ΔΔG     |          8754 single-point mutation from 301 proteins         |              [S8754.csv](data/ddG/S8754.csv)             | 2023-05 |
|         S783         |     ΔΔG     |           783 single-point mutation from 55 proteins          |               [S783.csv](data/ddG/S783.csv)              | 2023-05 |
|         S669         |     ΔΔG     |           669 single-point mutation from 94 proteins          |               [S669.csv](data/ddG/S669.csv)              | 2022-01 |
|         S461         |     ΔΔG     |           461 single-point mutation from 48 proteins          |               [S461.csv](data/ddG/S461.csv)              | 2023-01 |
|         S1626        |     ΔTm     |          1626 single-point mutation from 95 proteins          | This file is included in [S4346.csv](data/ddG/S4346.csv) | 2016-06 |
|         S4346        |     ΔTm     |          4346 single-point mutation from 349 proteins         |              [S4346.csv](data/dTm/S4346.csv)             | 2023-05 |
|         S571         |     ΔTm     |           571 single-point mutation from 37 proteins          |               [S571.csv](data/dTm/S571.csv)              | 2023-05 |


Other data files in the `data` folder

## Prerequisites

* download softwares
  * [AlphaFold](https://github.com/deepmind/alphafold)
  * [FoldX](https://foldxsuite.crg.eu)
  * [Anaconda](https://www.anaconda.com)

## Install software on Linux

1. download `GeoStab`

```bash
git clone https://github.com/Gonglab-THU/GeoStab.git
cd GeoStab
```

2. install `FoldX` software

3. install `Anaconda` software

4. install Python packages from `Anaconda`

```bash
conda create -n geostab python=3.10
conda activate geostab

conda install pytorch cpuonly -c pytorch
pip install biopandas
pip install biopython
pip install click
pip install pdb-tools
```

## Usage

* You should modify the mutant information in `generate_example_mut_info_csv.py` (0 <= pH <= 11 and 0 <= temperature <= 120)
* You should modify the `software_foldx` in `run.sh`
* You should install `AlphaFold2` and modify script in `run.sh` to generate `Seq` version structure

```bash
# fitness Seq prediction
python generate_example_mut_info_csv.py
bash run.sh -m fitness -v Seq -o ./example_Seq

# fitness 3D prediction
python generate_example_mut_info_csv.py
bash run.sh -m fitness -v threeD -o ./example_3D

# ΔΔG/ΔTm Seq prediction
python generate_example_mut_info_csv.py
bash run.sh -m ddGdTm -v Seq -o ./example_Seq

# ΔΔG/ΔTm 3D prediction
python generate_example_mut_info_csv.py
bash run.sh -m ddGdTm -v threeD -o ./example_3D
```

## Reference

[Improving the prediction of protein stability changes upon mutations by geometric learning and a pre-training strategy](https://doi.org/10.1038/s43588-024-00716-2)

## Acknowledgements

* [AlphaFold](https://github.com/deepmind/alphafold)
* [FoldX](https://foldxsuite.crg.eu)
