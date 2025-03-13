import os
import json
import pandas as pd

data = pd.read_csv('data/graph_data/graph/bace/bace.csv',index_col=0)

for split in ["train","valid","test"]:
    indices = data.loc[data['split']==split]["molecule_index"].to_list()
    # check if directory exists
    if not os.path.exists(f"experiments/raw_ofiicial_split/bace/{split}"):
        os.makedirs(f"experiments/raw_ofiicial_split/bace/{split}")
    with open(f"experiments/raw_ofiicial_split/bace/{split}/indices.json","w") as f:
        json.dump({"bace":indices},f)