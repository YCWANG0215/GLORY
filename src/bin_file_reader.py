import json
import pickle
from pathlib import Path
import torch

data_dir = {
    'test': 'data/MINDsmall/test',
    'train': 'data/MINDsmall/train',
    'val': 'data/MINDsmall/val',
}

def convert_to_readable_format(data):
    if isinstance(data, torch.Tensor):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: convert_to_readable_format(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_readable_format(item) for item in data]
    return data

sub_neighbors_output_path = Path(data_dir['train']) / "subcategory_neighbors.txt"
# sub_graph_output_path = Path(data_dir['train']) / "subcategory_graph.txt"

subcategory_neighbors_path = Path(data_dir['train']) / "subcategory_neighbor_dict.bin"
subcategory_neighbors = pickle.load(open(subcategory_neighbors_path, 'rb'))

readable_sub_neighbors = convert_to_readable_format(subcategory_neighbors)

with open(subcategory_neighbors_path, 'rb') as f:
    subcategory_neighbors = torch.load(f)
    readable_sub_graph = convert_to_readable_format(subcategory_neighbors)

with open(sub_neighbors_output_path, "w") as f:
    json.dump(readable_sub_neighbors, f, indent=4)

# with open(sub_graph_output_path, "w") as f:
#     json.dump(readable_sub_graph, f, indent=4)

# print(f"Data has been written to {output_txt_path}")
# print(subcategory_neighbors)