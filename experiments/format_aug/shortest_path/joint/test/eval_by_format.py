import json
import os
from collections import defaultdict



def main():
    # get the py file dir path 
    # py_file_dir = os.path.dirname(__file__)

    predictions = json.load(open(os.path.join(os.path.dirname(__file__),"instruction_dataset_with_prediction.json")))
    # formats = ["json", "graphml","gml","table"]
    num_pos = defaultdict(int)
    num_response = defaultdict(int)
    for entry in predictions:
        if entry.get("answer","ANSWER") == entry.get("predicted_answer","PREDICTED_ANSWER"):
            num_pos[entry["graph_format"]] += 1
        if "answer" in entry:
            num_response[entry["graph_format"]] += 1

    acc_200 = {k: v * 4 / len(predictions) for k, v in num_pos.items()}
    acc_ = {}
    for k,v in num_response.items():
        acc_[k] = num_pos[k] / v

    with open("acc.json", "w") as f:
        json.dump([acc_,acc_200], f, indent=4)


if __name__ == "__main__":
    main()
