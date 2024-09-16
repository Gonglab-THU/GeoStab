import os
import re
import pandas as pd
from tqdm import tqdm

#######################################################################
# predefined parameters
#######################################################################

os.system("mkdir -p dms_data")

amino_acid_list = list("ARNDCQEGHILKMFPSTWYV")
data_info = pd.read_csv("deep_sequence_data.csv", index_col=0)
data = pd.read_excel("41592_2018_138_MOESM4_ESM.xlsx", sheet_name=None)

#######################################################################
# process data
#######################################################################

for sheet_name in tqdm(data_info.index):

    os.system(f"mkdir -p dms_data/{sheet_name}")

    if sheet_name == "TPMT_HUMAN_Fowler2018":
        result = data[sheet_name].copy()
        result = result[~result["mutant"].str.contains("X")]
        result = result[~(result["mutant"] == "WT")]
    elif sheet_name == "PTEN_HUMAN_Fowler2018":
        result = data[sheet_name].copy()
        result = result[~result["mutant"].str.contains("X")]
        result = result[~(result["mutant"] == "WT")]
    elif sheet_name == "PA_FLU_Sun2015":
        result = data[sheet_name].copy()
        result = result[~result["mutant"].str.contains("X")]
        result = result[~result["mutant"].str.contains("_")]
    else:
        result = data[sheet_name].copy()

    result = result[result[data_info.loc[sheet_name, "label_name"]].notnull()]

    result = result.groupby("mutant")[data_info.loc[sheet_name, "label_name"]].mean().reset_index()
    result.columns = ["mut_info", "raw_label"]
    result.reset_index(inplace=True)
    result = result[["mut_info", "raw_label"]]

    tmp = {}
    seq = ""
    for multi_mut_info in result["mut_info"]:
        for single_mut_info in multi_mut_info.split(":"):
            a, b, c = single_mut_info[0], int(single_mut_info[1:-1]), single_mut_info[-1]
            assert a in amino_acid_list and c in amino_acid_list

            if b not in tmp.keys():
                tmp[b] = a
            else:
                assert tmp[b] == a

    for i in range(min(tmp.keys()), max(tmp.keys()) + 1):
        try:
            seq += tmp[i]
        except:
            seq += "."

    start, end = re.search(seq, data_info.loc[sheet_name, "uniprot_seq"]).span()

    diff = int(start - min(tmp.keys()))
    for i in range(len(result)):
        multi_mut_info = result.loc[i, "mut_info"]
        tmp = ""
        for single_mut_info in multi_mut_info.split(":"):
            a, b, c = single_mut_info[0], int(single_mut_info[1:-1]), single_mut_info[-1]
            tmp += a + str(int(b) + diff) + c + ","

        result.loc[i, "mut_info"] = tmp[:-1]

    result["raw_label"] = result["raw_label"].astype(float)

    result.set_index("mut_info", inplace=True)
    tmp = []
    for multi_mut_info in result.index:
        for single_mut_info in multi_mut_info.split(","):
            a, b, c = single_mut_info[0], int(single_mut_info[1:-1]), single_mut_info[-1]
            if a == c:
                if len(multi_mut_info.split(",")) > 1:
                    print(multi_mut_info)
                tmp.append(multi_mut_info)
                continue

    result.drop(tmp, inplace=True)
    result.to_csv(f"dms_data/{sheet_name}/label.csv")
