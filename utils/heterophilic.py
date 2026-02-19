# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

# pip install ogb
# pip install -U gdown scipy

import os
import os.path as osp
import numpy as np
import torch
import torch_geometric.transforms as T
from typing import Optional, Callable, List, Union, Dict, Any
from torch_geometric.datasets import HeterophilousGraphDataset, WikiCS
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils import remove_self_loops, from_networkx
from torch_geometric.utils.undirected import to_undirected
from torch_sparse import coalesce
import pandas as pd

from utils.classic import Planetoid
from definitions import ROOT_DIR

SPLITS_DIR = osp.join(ROOT_DIR, "splits")
CITYNETWORK_DATASETS = {"paris", "shanghai", "los_angeles", "london"}
CITYNETWORK_FOLDER_ALIASES = {
    "paris": ["paris"],
    "shanghai": ["shanghai"],
    "los_angeles": ["los_angeles", "los-angeles", "losangeles", "la"],
    "london": ["london"],
}
# prefer higher-hop labels if multiple label tensors exist
CITYNETWORK_LABEL_FILES = [
    "10-chunk_32-hop_node_labels.pt",
    "10-chunk_16-hop_node_labels.pt",
    "node_labels.pt",
]


def _normalize_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def _make_undirected_clean(data: Data) -> Data:
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_index = to_undirected(edge_index)
    edge_index, _ = coalesce(edge_index, None, data.num_nodes, data.num_nodes)
    data.edge_index = edge_index

    if getattr(data, "y", None) is not None and data.y is not None:
        if data.y.dim() > 1 and data.y.size(-1) == 1:
            data.y = data.y.view(-1)
    return data


def _apply_idx_split_as_masks(data: Data, split_idx: dict) -> Data:
    n = data.num_nodes
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[split_idx["train"]] = True
    val_mask[split_idx["valid"]]   = True
    test_mask[split_idx["test"]]   = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def get_fixed_splits(data: Data, dataset_name: str, seed: int) -> Data:
    """
    Robust split loader:
    1) If Data already has fold masks (N, K), pick column = seed.
       (Actor/film does this; some others may too.)
    2) If ogbn-arxiv, apply official OGB split.
    3) Else try to load splits/<dataset>_split_0.6_0.2_<seed>.npz
       If missing, create a random split and save it.
    """
    dataset_key = _normalize_name(dataset_name)

    # (1) Built-in k-fold masks on the Data object
    if hasattr(data, "train_mask") and data.train_mask is not None:
        if torch.is_tensor(data.train_mask) and data.train_mask.dim() == 2:
            # Masks can be [num_nodes, K] or [K, num_nodes] depending on dataset.
            if data.train_mask.size(0) == data.num_nodes:
                k = data.train_mask.size(1)
                col = int(seed) % k
                data.train_mask = data.train_mask[:, col]
                if torch.is_tensor(data.val_mask) and data.val_mask.dim() == 2:
                    data.val_mask = data.val_mask[:, col]
                if torch.is_tensor(data.test_mask) and data.test_mask.dim() == 2:
                    data.test_mask = data.test_mask[:, col]
            else:
                k = data.train_mask.size(0)
                col = int(seed) % k
                data.train_mask = data.train_mask[col]
                if torch.is_tensor(data.val_mask) and data.val_mask.dim() == 2:
                    data.val_mask = data.val_mask[col]
                if torch.is_tensor(data.test_mask) and data.test_mask.dim() == 2:
                    data.test_mask = data.test_mask[col]
            return data

    # (2) OGB official split
    if dataset_key in {"ogbn_arxiv", "ogbnarxiv", "ogbn_arxiv"} or dataset_name.lower() == "ogbn-arxiv":
        # We attach split_idx when building the dataset in get_dataset().
        if hasattr(data, "ogb_split_idx") and data.ogb_split_idx is not None:
            return _apply_idx_split_as_masks(data, data.ogb_split_idx)

        # Fallback: recreate split from OGB if attribute missing
        from ogb.nodeproppred import PygNodePropPredDataset
        ogb_root = osp.join(ROOT_DIR, "datasets")
        ogb_ds = PygNodePropPredDataset(name="ogbn-arxiv", root=ogb_root)
        split_idx = ogb_ds.get_idx_split()
        return _apply_idx_split_as_masks(data, split_idx)

    # (3) Repo-managed fixed splits
    os.makedirs(SPLITS_DIR, exist_ok=True)
    split_path = osp.join(SPLITS_DIR, f"{dataset_key}_split_0.6_0.2_{seed}.npz")

    try:
        with np.load(split_path) as f:
            train_mask = f["train_mask"]
            val_mask   = f["val_mask"]
            test_mask  = f["test_mask"]
    except FileNotFoundError:
        # Create and save deterministic random split
        n = data.num_nodes
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)

        n_train = int(0.6 * n)
        n_val   = int(0.2 * n)

        train_idx = perm[:n_train]
        val_idx   = perm[n_train:n_train + n_val]
        test_idx  = perm[n_train + n_val:]

        train_mask = np.zeros(n, dtype=bool)
        val_mask   = np.zeros(n, dtype=bool)
        test_mask  = np.zeros(n, dtype=bool)

        train_mask[train_idx] = True
        val_mask[val_idx]     = True
        test_mask[test_idx]   = True

        np.savez(split_path, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.val_mask   = torch.tensor(val_mask, dtype=torch.bool)
    data.test_mask  = torch.tensor(test_mask, dtype=torch.bool)

    # Keep your Planetoid special handling
    if dataset_key in {"cora", "citeseer", "pubmed"} and hasattr(data, "non_valid_samples"):
        data.train_mask[data.non_valid_samples] = False
        data.val_mask[data.non_valid_samples] = False
        data.test_mask[data.non_valid_samples] = False
    else:
        # Some datasets may not cover all nodes in masks; only assert when you know it's full coverage.
        pass

    return data


def _pick_first(keys, container):
    for key in keys:
        if key in container:
            return key
    return None


def _to_tensor(val, dtype=None):
    if val is None:
        return None
    if torch.is_tensor(val):
        return val.clone().to(dtype=dtype) if dtype is not None else val.clone()
    if isinstance(val, np.ndarray):
        t = torch.from_numpy(val)
        return t.to(dtype=dtype) if dtype is not None else t
    return torch.tensor(val, dtype=dtype)


def _edge_index_from_value(val):
    if val is None:
        return None
    if torch.is_tensor(val):
        ei = val.clone().long()
    else:
        arr = val
        if isinstance(arr, np.ndarray):
            pass
        else:
            arr = np.array(arr)
        if arr.ndim == 2 and arr.shape[0] == 2:
            ei = torch.tensor(arr, dtype=torch.long)
        elif arr.ndim == 2 and arr.shape[1] == 2:
            ei = torch.tensor(arr.T, dtype=torch.long)
        elif arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            row, col = np.nonzero(arr)
            ei = torch.tensor(np.vstack([row, col]), dtype=torch.long)
        else:
            raise ValueError("Edge data must be [2, E], [E, 2], or adjacency matrix.")
    if ei.dim() != 2 or ei.size(0) != 2:
        raise ValueError("edge_index tensor must have shape [2, E].")
    return ei


def _data_from_city_payload(payload: Dict[str, Any]):
    feat_key = _pick_first(["x", "feat", "feats", "features"], payload)
    y_key = _pick_first(["y", "label", "labels", "targets"], payload)
    edge_key = _pick_first(["edge_index", "edges", "edgeindex"], payload)
    adj_key = _pick_first(["adj", "adjacency", "A", "graph"], payload)

    if feat_key is None or y_key is None:
        raise ValueError("CityNetwork payload must include node features and labels.")

    x = _to_tensor(payload[feat_key], dtype=torch.float32)
    y = _to_tensor(payload[y_key]).long().view(-1)

    if edge_key is not None:
        edge_index = _edge_index_from_value(payload[edge_key])
    elif adj_key is not None:
        edge_index = _edge_index_from_value(payload[adj_key])
    else:
        raise ValueError("CityNetwork payload must include edge_index or adjacency.")

    data_kwargs = {"x": x, "y": y, "edge_index": edge_index}

    for key, value in payload.items():
        low = key.lower()
        if key in {feat_key, y_key, edge_key, adj_key}:
            continue
        if "mask" in low:
            data_kwargs[key] = _to_tensor(value, dtype=torch.bool)
        elif isinstance(value, (list, tuple, np.ndarray)) or torch.is_tensor(value):
            try:
                data_kwargs[key] = _to_tensor(value)
            except Exception:
                pass

    return Data(**data_kwargs)


def _load_city_network_pt(path: str) -> Data:
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, Data):
        return obj

    if isinstance(obj, (list, tuple)):
        for item in obj:
            if isinstance(item, Data):
                return item
            if isinstance(item, dict):
                return _data_from_city_payload(item)

    if isinstance(obj, dict):
        return _data_from_city_payload(obj)

    if hasattr(obj, "__len__"):
        try:
            first = obj[0]
            if isinstance(first, Data):
                return first
            if isinstance(first, dict):
                return _data_from_city_payload(first)
        except Exception:
            pass

    raise TypeError(f"Unrecognized city dataset format in '{path}'.")


def _load_city_network_npz(path: str) -> Data:
    with np.load(path, allow_pickle=True) as npz_file:
        payload = {k: npz_file[k] for k in npz_file.files}
    return _data_from_city_payload(payload)


def _city_dir_candidates(name_key: str):
    aliases = CITYNETWORK_FOLDER_ALIASES.get(name_key, [])
    # Always include the raw key as the final fallback.
    if name_key not in aliases:
        aliases = list(aliases) + [name_key]
    return aliases


def _load_city_network_dir(path: str) -> Data:
    cache = {}

    def _maybe_load(fname: str):
        if fname in cache:
            return cache[fname]
        fpath = osp.join(path, fname)
        if osp.exists(fpath):
            cache[fname] = torch.load(fpath, map_location="cpu", weights_only=False)
        else:
            cache[fname] = None
        return cache[fname]

    # prefer augmented node features when present
    x_aug = _maybe_load("node_features_augmented.pt")
    x_base = _maybe_load("node_features.pt")
    features = x_aug if x_aug is not None else x_base
    if features is None:
        raise FileNotFoundError(f"Missing node_features.pt in {path}")

    label_tensor = None
    label_file = None
    for candidate in CITYNETWORK_LABEL_FILES:
        tensor = _maybe_load(candidate)
        if tensor is not None:
            label_tensor = tensor
            label_file = candidate
            break
    if label_tensor is None:
        # Fall back to any file that contains "label".
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".pt") and "label" in fname.lower():
                tensor = _maybe_load(fname)
                if tensor is not None:
                    label_tensor = tensor
                    label_file = fname
                    break
    if label_tensor is None:
        raise FileNotFoundError(f"Missing node label tensor in {path}")

    edge_index_tensor = _maybe_load("edge_indices.pt")
    if edge_index_tensor is None:
        raise FileNotFoundError(f"Missing edge_indices.pt in {path}")

    payload = {
        "features": features,
        "targets": label_tensor,
        "edge_index": edge_index_tensor,
    }
    data = _data_from_city_payload(payload)

    # attach masks
    mask_files = {
        "train_mask.pt": "train_mask",
        "valid_mask.pt": "val_mask",
        "val_mask.pt": "val_mask",
        "test_mask.pt": "test_mask",
    }
    for fname, attr in mask_files.items():
        tensor = _maybe_load(fname)
        if tensor is not None:
            setattr(data, attr, _to_tensor(tensor, dtype=torch.bool).view(-1))

    # attach alternate feature variants for convenience
    if x_base is not None:
        data.x_raw = _to_tensor(x_base, dtype=torch.float32)
    if x_aug is not None:
        data.x_augmented = _to_tensor(x_aug, dtype=torch.float32)

    # attach edge attributes if provided
    edge_attr = _maybe_load("edge_features.pt")
    if edge_attr is not None:
        data.edge_attr = _to_tensor(edge_attr, dtype=torch.float32)

    # attach hop eccentricities if available
    for hop in (16, 32):
        tensor = _maybe_load(f"{hop}-hop_eccentricities.pt")
        if tensor is not None:
            setattr(data, f"eccentricity_{hop}", _to_tensor(tensor, dtype=torch.float32))

    # store alternative label tensors if they exist so downstream code can switch if desired
    for candidate in CITYNETWORK_LABEL_FILES:
        if candidate == label_file:
            continue
        tensor = _maybe_load(candidate)
        if tensor is not None:
            suffix = candidate.replace("-", "_").replace(".", "_")
            attr = f"labels_{suffix}"
            setattr(data, attr, _to_tensor(tensor).long().view(-1))

    return data


def load_city_network_dataset(name: str):
    name_key = _normalize_name(name)
    if name_key not in CITYNETWORK_DATASETS:
        raise ValueError(f"Unknown CityNetwork dataset '{name}'.")

    city_root = osp.join(ROOT_DIR, "datasets", "city_networks")
    os.makedirs(city_root, exist_ok=True)
    candidates = [
        osp.join(city_root, f"{name_key}.pt"),
        osp.join(city_root, f"{name_key}.pth"),
        osp.join(city_root, f"{name_key}.npz"),
    ]
    dir_candidates = [
        osp.join(city_root, candidate) for candidate in _city_dir_candidates(name_key)
    ]
    data_path = next((path for path in candidates if osp.exists(path)), None)
    dir_path = next((path for path in dir_candidates if osp.isdir(path)), None)

    if data_path is None and dir_path is None:
        expected = ", ".join(candidates + dir_candidates)
        raise FileNotFoundError(
            f"CityNetwork dataset '{name}' not found. "
            f"Place a file named {name_key}.pt/.pth/.npz or a folder under {city_root}. "
            f"Checked: {expected}"
        )

    if data_path is not None:
        if data_path.endswith(".npz"):
            data = _load_city_network_npz(data_path)
        else:
            data = _load_city_network_pt(data_path)
    else:
        data = _load_city_network_dir(dir_path)

    if isinstance(data, list):
        cleaned = []
        for item in data:
            if isinstance(item, Data):
                cleaned.append(_make_undirected_clean(item))
            else:
                cleaned.append(_make_undirected_clean(_data_from_city_payload(item)))
        return cleaned

    data = _make_undirected_clean(data)
    return [data]


def get_synthetic_dataset(name, args):
    data_root = osp.join(ROOT_DIR, 'datasets')
    # Getting the Synthetic Dataset. 
    dataset = SyntheticData(data_root,name, args)
    return dataset




def apply_ogbn_arxiv_split(data, ogb_dataset):
    '''
        Applies the standard OGBN-ArXiv split to the given data object.
    '''
    split_idx = ogb_dataset.get_idx_split()  # dict with train/valid/test tensors
    n = data.num_nodes
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[split_idx['train']] = True
    val_mask[split_idx['valid']]   = True
    test_mask[split_idx['test']]   = True
    data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
    return data

def load_arxiv_year_via_ogb(root: str):
    """
    Builds arxiv-year from ogbn-arxiv: label = node_year (commonly used setup). :contentReference[oaicite:4]{index=4}
    Returns a list [Data] to match your convention.
    """
    from ogb.nodeproppred import PygNodePropPredDataset

    ogb_ds = PygNodePropPredDataset(name="ogbn-arxiv", root=root)
    data = ogb_ds[0]

    if not hasattr(data, "node_year"):
        raise AttributeError(
            "ogbn-arxiv Data has no 'node_year' attribute in your OGB/PyG version. "
            "Print(data) to inspect available fields."
        )

    data.y = data.node_year.view(-1).long()
    data = _make_undirected_clean(data)

    # You can still store the official split idx (even if labels changed)
    data.ogb_split_idx = ogb_ds.get_idx_split()
    return [data]

def get_dataset(name: str):
    """
    Returns either:
    - a PyG InMemoryDataset-like object (so dataset[0] is Data), OR
    - a list [Data] (for your special cases)
    """
    data_root = osp.join(ROOT_DIR, "datasets")
    name_key = _normalize_name(name)
    if name_key == "la":
        name_key = "los_angeles"

    # --- WebKB ---
    if name_key in {"cornell", "texas", "wisconsin"}:
        dataset = WebKB(root=data_root, name=name_key, transform=T.NormalizeFeatures())

    # --- WikipediaNetwork ---
    elif name_key in {"chameleon", "squirrel"}:
        dataset = WikipediaNetwork(root=data_root, name=name_key, transform=T.NormalizeFeatures())

    # --- Actor / Film (aliases) ---
    elif name_key in {"film", "actor"}:
        # Your Actor class is actually the "film" dataset
        dataset = Actor(root=data_root, transform=T.NormalizeFeatures())

    # --- Planetoid ---
    elif name_key in {"cora", "citeseer", "pubmed"}:
        dataset = Planetoid(root=data_root, name=name_key, transform=T.NormalizeFeatures())

    # --- Wiki-CS ---
    elif name_key in {"wiki_cs", "wikics", "wiki-cs"}:
        dataset = WikiCS(root=data_root, transform=T.NormalizeFeatures())
        dataset.data = _make_undirected_clean(dataset.data)

    # --- OGBN-ArXiv (auto-download via OGB, no local .pt required) ---
    elif name_key in {"ogbn_arxiv", "ogbnarxiv"} or name.lower() == "ogbn-arxiv":
        from ogb.nodeproppred import PygNodePropPredDataset
        ogb_dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=data_root)
        data = ogb_dataset[0]
        data.y = data.y.view(-1)
        # store official split for get_fixed_splits()
        data.ogb_split_idx = ogb_dataset.get_idx_split()
        # ONLY do this if your Laplacian/code assumes undirected:
        data = _make_undirected_clean(data)
        dataset = [data]  # keep your “list-like” convention
 
     # --- arxiv-year (OGB-based build) ---
    elif name_key in {"arxiv_year", "arxivyear", "arxiv-year"}:
        dataset = load_arxiv_year_via_ogb(data_root)

    # --- snap-patents (Drive .mat download + convert) ---
    elif name_key in {"snap_patents", "snappatents", "snap-patents"}:
        dataset = SnapPatents(
            root=osp.join(data_root, "snap_patents"),
            transform=T.NormalizeFeatures(),
        )
        # dataset.data already cleaned in process()

    elif name_key in CITYNETWORK_DATASETS:
        dataset = load_city_network_dataset(name_key)

    # --- Roman Empire (local files) ---
    elif name_key in {"roman_empire", "romanempire"}:
        dataset = HeterophilousGraphDataset(
            root=osp.join(data_root, "heterophilous"),
            name="roman_empire",
            transform=T.NormalizeFeatures(),
        )
        dataset.data = _make_undirected_clean(dataset.data)  # if your laplacian assumes undirected

    elif name_key in {
        "amazon_ratings", "amazonratings", "amazon-ratings",
        "amazon_rating", "amazonrating"
    }:
        dataset = HeterophilousGraphDataset(
            root=osp.join(data_root, "heterophilous"),
            name="amazon-ratings",               # NOTE: PyG canonical name
            transform=T.NormalizeFeatures(),
        )
        dataset.data = _make_undirected_clean(dataset.data)

    # --- Tokyo Dataset (local files, custom loader) ---
    elif name_key in {"tokyo_railway"}:
        dataset = TokyoRailway(root=data_root, name=name_key)
    else:
        raise ValueError(f"dataset {name} not supported in dataloader")

    return dataset


class Actor(InMemoryDataset):
    r"""
    Code adapted from https://github.com/pyg-team/pytorch_geometric/blob/2.0.4/torch_geometric/datasets/actor.py

    The actor-only induced subgraph of the film-director-actor-writer
    network used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Each node corresponds to an actor, and the edge between two nodes denotes
    co-occurrence on the same Wikipedia page.
    Node features correspond to some keywords in the Wikipedia pages.
    The task is to classify the nodes into five categories in term of words of
    actor's Wikipedia.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> List[str]:
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt'
                ] + [f'film_split_0.6_0.2_{i}.npz' for i in range(10)]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        for f in self.raw_file_names[:2]:
            download_url(f'{self.url}/new_data/film/{f}', self.raw_dir)
        for f in self.raw_file_names[2:]:
            download_url(f'{self.url}/splits/{f}', self.raw_dir)

    def process(self):

        with open(self.raw_paths[0], 'r') as f:
            data = [x.split('\t') for x in f.read().split('\n')[1:-1]]

            rows, cols = [], []
            for n_id, col, _ in data:
                col = [int(x) for x in col.split(',')]
                rows += [int(n_id)] * len(col)
                cols += col
            x = SparseTensor(row=torch.tensor(rows), col=torch.tensor(cols))
            x = x.to_dense()

            y = torch.empty(len(data), dtype=torch.long)
            for n_id, _, label in data:
                y[int(n_id)] = int(label)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            # Remove self-loops
            edge_index, _ = remove_self_loops(edge_index)
            # Make the graph undirected
            edge_index = to_undirected(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        train_masks, val_masks, test_masks = [], [], []
        for f in self.raw_paths[2:]:
            tmp = np.load(f)
            train_masks += [torch.from_numpy(tmp['train_mask']).to(torch.bool)]
            val_masks += [torch.from_numpy(tmp['val_mask']).to(torch.bool)]
            test_masks += [torch.from_numpy(tmp['test_mask']).to(torch.bool)]
        train_mask = torch.stack(train_masks, dim=1)
        val_mask = torch.stack(val_masks, dim=1)
        test_mask = torch.stack(test_masks, dim=1)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])


class WikipediaNetwork(InMemoryDataset):
    r"""
    Code adapted from https://github.com/pyg-team/pytorch_geometric/blob/2.0.4/torch_geometric/datasets/wikipedia_network.py

    The Wikipedia networks introduced in the
    `"Multi-scale Attributed Node Embedding"
    <https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features represent several informative nouns in the Wikipedia pages.
    The task is to predict the average daily traffic of the web page.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"chameleon"`,
            :obj:`"crocodile"`, :obj:`"squirrel"`).
        geom_gcn_preprocess (bool): If set to :obj:`True`, will load the
            pre-processed data as introduced in the `"Geom-GCN: Geometric
            Graph Convolutional Networks" <https://arxiv.org/abs/2002.05287>_`,
            in which the average monthly traffic of the web page is converted
            into five categories to predict.
            If set to :obj:`True`, the dataset :obj:`"crocodile"` is not
            available.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    """
    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        assert self.name in ['chameleon', 'squirrel']
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self.url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/new_data')

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str]]:
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        # downloads:
        # new_data/<name>/out1_node_feature_label.txt
        # new_data/<name>/out1_graph_edges.txt
        for f in self.raw_file_names:
            download_url(f"{self.url}/{self.name}/{f}", self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
        x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
        x = torch.tensor(x, dtype=torch.float)
        y = [int(r.split('\t')[2]) for r in data]
        y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
        edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
        # Remove self-loops
        edge_index, _ = remove_self_loops(edge_index)
        # Make the graph undirected
        edge_index = to_undirected(edge_index)
        edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])


class WebKB(InMemoryDataset):
    r"""
    Code adapted from https://github.com/pyg-team/pytorch_geometric/blob/2.0.4/torch_geometric/datasets/webkb.py

    The WebKB datasets used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features are the bag-of-words representation of web pages.
    The task is to classify the nodes into one of the five categories, student,
    project, course, staff, and faculty.
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cornell"`,
            :obj:`"Texas"`, :obj:`"Wisconsin"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/'
           '1c4c04f93fa6ada91976cda8d7577eec0e3e5cce/new_data')

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cornell', 'texas','wisconsin']
        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float32)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index = to_undirected(edge_index)
            # We also remove self-loops in these datasets in order not to mess up with the Laplacian.
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)





class SyntheticData(InMemoryDataset):
    
    def __init__(self, root, name, args, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['synthetic_exp']
        self.num_nodes = args.num_nodes
        self.n_classes = args.num_classes
        self.num_feats = args.num_feats
        self.het = args.het_coef
        self.p = 1-args.edge_noise
        self.K = args.node_degree
        self.r = args.ellipsoid_radius
        self.feat_noise = args.feat_noise
        self.just_add_noise = args.just_add_noise
        self.ellipsoids = args.ellipsoids
        self.matriu_corr = None
        if args.classes_corr is not None:
            n_classes = int(np.sqrt(len(args.classes_corr)))
            self.matriu_corr = np.zeros((n_classes,n_classes))
            for i in range(n_classes):
                for j in range(n_classes):
                    self.matriu_corr[i][j] = args.classes_corr[i*n_classes+j]
        else:
            self.matriu_corr = self.het*(1/(self.n_classes-1))*np.ones((self.n_classes,self.n_classes))
            for i in range(self.n_classes):
                self.matriu_corr[i][i] = 1-self.het
        super(SyntheticData, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return 'synthetic_data.pt'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def generate_features(self):
        y = torch.ones((self.num_nodes))
        for i in range(self.num_nodes):
            y[i] = np.random.randint(0,self.n_classes)
        y = y.type(torch.LongTensor)
        if self.ellipsoids == True:
            class_params = torch.tensor(np.random.rand(self.n_classes,self.num_feats))
            mean_class_params = torch.mean(class_params,dim=0)
            torch.save(class_params,self.raw_dir+'/class_params.pt')
            x = torch.ones((self.num_nodes,self.num_feats),dtype=float)
            x_coords = np.pi*np.random.rand(self.num_nodes,self.num_feats-1)
            x_coords[:][-1] = x_coords[:][-1]*2
            noise = torch.normal(mean=torch.zeros(self.num_nodes,self.num_feats,dtype=float),std=self.feat_noise)
            for i in range(self.num_nodes):
                x[i] = class_params[y[i]]*self.r*x[i]
                #we also rescale the noise to make it a percentage
                noise[i] = class_params[y[i]]*noise[i]
                for j in range(self.num_feats-1):
                    x[i][j] = np.cos(x_coords[i][j])*x[i][j]
                    for k in range(j+1,self.num_feats):
                        x[i][k] = np.sin(x_coords[i][j])*x[i][k]
            x = x+noise
        else:
            y = torch.ones((self.num_nodes))
            for i in range(self.num_nodes):
                y[i] = np.random.randint(0,self.n_classes)
            y = torch.tensor(y).detach()
            y = y.type(torch.LongTensor)
            #generate random multivariate normal distributions
            center_means = torch.tensor(np.random.rand(self.num_feats),dtype=float)
            means = torch.tensor(np.random.rand(self.n_classes,self.num_feats),dtype=float)*0.25
            cov_mat = torch.tensor(np.random.rand(self.num_feats,self.num_feats),dtype=float)
            cov_mat = torch.matmul(torch.transpose(cov_mat,-2,-1),cov_mat)
            for i in range(self.n_classes):
                means[i] = center_means+means[i]
                

            x = torch.ones((self.num_nodes,self.num_feats))

            for i in range(x.shape[0]):
                act_class = y[i]
                distrib = torch.distributions.multivariate_normal.MultivariateNormal(means[act_class], cov_mat)
                x[i] = distrib.rsample()
        x = x.type(torch.FloatTensor)
        return x, y
    
    def generate_edges(self, y):

        no_edge = [[] for l in range(self.num_nodes)]
        
        for i in range(self.num_nodes):
            no_edge[i] = [ [] for l in range(self.n_classes)]
            for j in range(self.num_nodes):
                if 0 == abs(i-j)%self.num_nodes or abs(i-j) > self.K/2:
                    no_edge[i][y[j]].append(j)
        edge_index = [[],[]]
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if 0 < (j-i)%self.num_nodes and (j-i)%self.num_nodes <= self.K/2 and np.random.random() <= self.p:
                    ck = int(np.random.choice(range(self.n_classes),1,p=self.matriu_corr[y[i]]))
                    while len(no_edge[i][ck]) == 0:
                        ck = (ck+1)%self.n_classes
                    ind_k = np.random.randint(0,len(no_edge[i][ck]))
                    k = no_edge[i][ck][ind_k]
                    edge_index[0] = edge_index[0]+[i,k]
                    edge_index[1] = edge_index[1]+[k,i]
                    no_edge[i][ck].pop(ind_k)
                elif 0 < (j-i)%self.num_nodes and (j-i)%self.num_nodes <= self.K/2:
                    edge_index[0] = edge_index[0]+[i,j]
                    edge_index[1] = edge_index[1]+[j,i]
        return edge_index
    
    def download(self):
        matriu_corr = self.matriu_corr
        #we try to read from data file the features, for fair comparison
        try:
            x = torch.load(self.raw_dir+'/x_data.pt', weights_only=False)
            y = torch.load(self.raw_dir+'/y_data.pt', weights_only=False)
            if self.just_add_noise:
                class_params = torch.load(self.raw_dir+'/class_params.pt', weights_only=False)
                mean_class_params = torch.mean(class_params,dim=0)
                noise = torch.normal(mean=torch.zeros(self.num_nodes,
                                                      self.num_feats,
                                                      dtype=float),
                                     std=self.feat_noise)
                for i in range(self.num_nodes):
                    noise[i] = mean_class_params*noise[i]
                x = x+noise
                x = x.type(torch.FloatTensor)
        except:
            x, y = self.generate_features()
            torch.save(x,self.raw_dir+'/x_data.pt')
            torch.save(y,self.raw_dir+'/y_data.pt')
        try:
            edge_index = torch.load(self.raw_dir+'/edge_data.pt', weights_only=False)
        except:
            edge_index = self.generate_edges(y)
            torch.save(edge_index,self.raw_dir+'/edge_data.pt')
        data = [Data(x=x,edge_index = torch.tensor(edge_index),y = y)]
        torch.save(self.collate(data),self.raw_dir+'/synthetic_data.pt')

    def process(self):
        data = torch.load(self.raw_dir+'/synthetic_data.pt', weights_only=False)
        x = data[0].x
        y = data[0].y

        edge_index = data[0].edge_index
        # We also remove self-loops in these datasets in order not to mess up with the Laplacian.
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


class SnapPatents(InMemoryDataset):
    """
    Downloads snap_patents.mat (Google Drive) and converts to a PyG Data object.

    Source is the commonly used Non-Homophily-Large-Scale benchmark assets. :contentReference[oaicite:2]{index=2}
    """
    # This is the Google Drive "uc?id=" link corresponding to snap_patents.mat in that benchmark README.
    # If the file ever moves, update this ID based on the README source.
    gdrive_url = "https://drive.google.com/uc?id=1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia"  # snap_patents.mat :contentReference[oaicite:3]{index=3}

    def __init__(self, root: str, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ["snap_patents.mat"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        out_path = self.raw_paths[0]

        # Use gdown because Drive links are not direct-file URLs.
        try:
            import gdown
        except ImportError as e:
            raise ImportError(
                "snap-patents download requires gdown. Install with: pip install gdown"
            ) from e

        gdown.download(self.gdrive_url, out_path, quiet=False)

    def process(self):
        try:
            import scipy.io as sio
        except ImportError as e:
            raise ImportError(
                "snap-patents processing requires scipy. Install with: pip install scipy"
            ) from e

        mat = sio.loadmat(self.raw_paths[0])

        # ---- Robust key discovery (mat files vary across repos) ----
        def pick_first(keys):
            for k in keys:
                if k in mat:
                    return k
            return None

        x_key = pick_first(["x", "features", "node_features", "node_feat", "node_feat"])  # ok
        y_key = pick_first(["y", "labels", "label", "node_labels", "years", "year", "node_year"])  # <-- ADD THESE
        adj_key = pick_first(["adj", "A", "network", "edge_index"])  # ok

        if x_key is None or y_key is None or adj_key is None:
            raise KeyError(
                f"Could not find required keys in .mat. Found keys={list(mat.keys())}. "
                f"Need one of x={['x','features','node_features','node_feat']}, "
                f"y={['y','labels','label','node_labels','years','year','node_year']}, "
                f"adj={['adj','A','network','edge_index']}."
            )

        X = mat[x_key]
        Y = mat[y_key]
        A = mat[adj_key]

        # ---- X -> tensor ----
        try:
            import scipy.sparse as sp
            if sp.issparse(X):
                x = torch.from_numpy(X.tocsr().astype(np.float32).toarray())
            else:
                x = torch.tensor(X, dtype=torch.float32)
        except Exception:
            x = torch.tensor(X, dtype=torch.float32)

        # ---- Y -> contiguous class ids ----
        y_raw = torch.tensor(np.array(Y).squeeze(), dtype=torch.long)

        # IMPORTANT: remap arbitrary years (e.g., 1976..2017) to 0..C-1
        # so CrossEntropyLoss works properly.
        _, y = torch.unique(y_raw, sorted=True, return_inverse=True)

        # ---- edge_index ----
        if isinstance(A, np.ndarray) and A.ndim == 2 and A.shape[0] == 2:
            edge_index = torch.tensor(A, dtype=torch.long)
        else:
            # if it is sparse adjacency (not your case right now)
            import scipy.sparse as sp
            if not sp.issparse(A):
                A = sp.coo_matrix(A)
            A = A.tocoo()
            edge_index = torch.stack(
                [torch.from_numpy(A.row).long(), torch.from_numpy(A.col).long()], dim=0
            )

        data = Data(x=x, edge_index=edge_index, y=y)
        data = _make_undirected_clean(data)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

class TokyoRailway(InMemoryDataset):
    """
    Loads the Tokyo Railway dataset from local .pt files. Expects a folder named "tokyo_railway" under datasets/
    """

    def __init__(self, root: str, name, transform=None, pre_transform=None):
        self.name = name.lower()
        super(TokyoRailway, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')
    
    @property
    def raw_file_names(self):
        return ["line_station_connectionV1130.csv", "pass_survey_tokyov1109.csv"]

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        # No downloading since we expect local files. Just check existence.
        for fname in self.raw_file_names:
            path = osp.join(self.raw_dir, fname)
            if not osp.exists(path):
                raise FileNotFoundError(f"Expected file {fname} not found in {self.raw_dir}. Please place it there.")

    def process(self):
        # Load required tensors
        connection_pd = pd.read_csv(self.raw_paths[0])
        connection_pd
        
        # get station list from connection_pd (both station_cd1 and station_cd2) to ensure we have all stations
        station_list_connection_1 = connection_pd[['station_cd1', 'station_name1']].rename(columns={'station_cd1': 'station_id', 'station_name1': 'S12_001'}).drop_duplicates()
        station_list_connection_2 = connection_pd[['station_cd2', 'station_name2']].rename(columns={'station_cd2': 'station_id', 'station_name2': 'S12_001'}).drop_duplicates()
        station_list_pd = pd.concat([station_list_connection_1, station_list_connection_2]).drop_duplicates().reset_index(drop=True)

        # list of all the stations in the passenger survey data that are also in the connection data
        passenger_survey_pd = pd.read_csv(self.raw_paths[1])
        passenger_survey_pd['station_id'] = passenger_survey_pd['station_id'].astype(int)
        passenger_survey_pd[passenger_survey_pd['station_id'].isin(station_list_pd['station_id'].values)]

        matching_passenger_survey_pd = passenger_survey_pd[passenger_survey_pd['station_id'].isin(station_list_pd['station_id'].values)].reset_index(drop=True)
        matching_passenger_survey_pd = matching_passenger_survey_pd.sort_values('station_id').reset_index(drop=True)

        # max-min normalize passenger counts:
        year_cols = ['2013', '2014', '2015', '2016', '2017', '2018', '2019']

        # normalization with global values:
        global_min = matching_passenger_survey_pd[year_cols].min().min()
        global_max = matching_passenger_survey_pd[year_cols].max().max()
        for year in year_cols:
            matching_passenger_survey_pd[year] = (matching_passenger_survey_pd[year] - global_min) / (global_max - global_min)

        # COMMENTED OUT FOR EXPERIMENTATION:
        # normalization with row values: 
        # row_mins = matching_passenger_survey_pd[year_cols].min(axis=1)
        # row_maxs = matching_passenger_survey_pd[year_cols].max(axis=1)
        # for year in year_cols:
        #     matching_passenger_survey_pd[year] = (matching_passenger_survey_pd[year] - row_mins) / (row_maxs - row_mins)

        # and now we need to make sure nodes in the passenger survey match up with nodes in the connection data
        # the passenger_survey_pd only has nodes that are in the station_list_pd, so we can just filter the station_list_pd to only include those nodes
        # note that both data had stations that were not in the other
        station_list_pd = station_list_pd[station_list_pd['station_id'].isin(matching_passenger_survey_pd['station_id'].values)].reset_index(drop=True)
        station_list_pd = station_list_pd.sort_values('station_id').reset_index(drop=True)

        station_node_pd = connection_pd[connection_pd['station_cd1'].isin(station_list_pd['station_id'].values) & connection_pd['station_cd2'].isin(station_list_pd['station_id'].values)][['line', 'station_cd1', 'station_cd2', 'distance']].drop_duplicates()

        # build graph structure using networkx
        import networkx as nx
        G = nx.Graph()
        G = nx.from_pandas_edgelist(station_node_pd, 'station_cd1', 'station_cd2', edge_attr=True)
        
        node_order = list(G.nodes())
        print(f"Number of nodes in G: {len(G.nodes())}")
        print(f"Number of edges in G: {len(G.edges())}")

        # we need to make sure the station_list_pd is in the same order as the nodes in the graph G, which is the order the adjacency matrix will use
        station_list_pd = station_list_pd.set_index('station_id').loc[node_order].reset_index(drop=False)
        station_list_pd

        # and we also need to make sure the passenger survey data is in the same order as the nodes in the graph G, which is the order the adjacency matrix will use
        matching_passenger_survey_pd = matching_passenger_survey_pd.set_index('station_id').loc[node_order].reset_index(drop=False)

        # get operator form line name
        station_node_pd['operator'] = station_node_pd['line'].apply(lambda x: x.split('.')[0])

        # Aggregate passenger survey by station_id (handles duplicate rows per station)
        survey_filtered = matching_passenger_survey_pd[matching_passenger_survey_pd['station_id'].isin(node_order)]
        survey_agg = survey_filtered.groupby('station_id')[['2013','2014','2015','2016','2017','2018','2019']].mean()

        #one hot encoding for line and operator
        all_operators = sorted(station_node_pd['operator'].unique())
        all_lines = sorted(station_node_pd['line'].unique())

        node_operator_df = pd.DataFrame(0, index=node_order, columns=all_operators)
        node_line_df = pd.DataFrame(0, index=node_order, columns=all_lines)

        for _, row in station_node_pd.iterrows():
            for node_col in ['station_cd1', 'station_cd2']:
                node = row[node_col]
                if node in node_operator_df.index:
                    node_operator_df.loc[node, row['operator']] = 1
                if node in node_line_df.index:
                    node_line_df.loc[node, row['line']] = 1

        operator_features = torch.tensor(node_operator_df.values, dtype=torch.float)
        line_features = torch.tensor(node_line_df.values, dtype=torch.float)

        # Reorder features to match graph node ordering

        #Previously, without masking-type learning:
        # train_x = torch.tensor(survey_agg.loc[node_order][['2013', '2014', '2015', '2016', '2017']].to_numpy(), dtype=torch.float)
        # train_x = torch.cat([train_x, operator_features, line_features], dim=1)  # Concatenate all features
        # train_y = torch.tensor(survey_agg.loc[node_order][['2018']].to_numpy(), dtype=torch.float)

        # Now, with masking to suit sheaf learning procedure:
        x = torch.tensor(survey_agg.loc[node_order][['2013', '2014', '2015', '2016', '2017', '2018']].to_numpy(), dtype=torch.float)
        x = torch.cat([x, operator_features, line_features], dim=1)  # Concatenate all features
        y = torch.tensor(survey_agg.loc[node_order][['2019']].to_numpy(), dtype=torch.float)
        data = from_networkx(G)
        data.x = x
        data.y = y
        data = _make_undirected_clean(data)
        data = data if self.pre_transform is None else self.pre_transform(data)
        print(f"Final data object: {data}")
        torch.save(self.collate([data]), self.processed_paths[0])

# if __name__ == "__main__":
#     # Example usage:
#     dataset = TokyoRailway(root="datasets", name="tokyo_railway")
#     print(dataset[0])
#     print(dataset[0].x)