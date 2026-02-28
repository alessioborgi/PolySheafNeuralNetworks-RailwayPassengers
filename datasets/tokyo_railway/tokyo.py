import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import numpy as np
import pandas as pd
import networkx as nx

import random
import time

import argparse
import os

from exp.run import aget
from utils.node_subsets import build_node_subset_masks

class GCN(nn.Module):
    def __init__(self, in_features, out_features, adjacency_matrix):
        super(GCN, self).__init__()
        A = adjacency_matrix
        A = A + torch.eye(A.size(0))  # Add self-loops
        D = torch.diag(torch.sum(A.abs(), dim=1))  # Use |A| for degree to handle negative weights
        D_inv_sqrt = torch.sqrt(torch.inverse(D))
        self.A_hat = torch.matmul(torch.matmul(D_inv_sqrt, A), D_inv_sqrt)  # Symmetric normalization
        self.layer_1 = nn.Linear(in_features, in_features)
        self.layer_2 = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        x = torch.matmul(self.A_hat, x)
        x = self.layer_1(x)
        x = torch.relu(x)
        x = torch.matmul(self.A_hat, x)
        x = self.layer_2(x)
        return x


class DenseGATLayer(nn.Module):
    """A single dense GAT attention head with valued masking.
    
    Instead of binary masking (on/off), the continuous adjacency weights
    scale the learned attention coefficients post-softmax:
        alpha'_ij = softmax(e_ij) * A_ij
    This lets distance/correlation values modulate how much each
    neighbor's message is trusted, while the network still learns
    *who* to attend to via standard GAT attention.
    """
    def __init__(self, in_features, out_features, adjacency_matrix, dropout=0.6, alpha=0.2):
        super(DenseGATLayer, self).__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        # Attention vector a splits into [a_left | a_right], each of size out_features
        self.a_left = nn.Parameter(torch.zeros(out_features, 1))
        self.a_right = nn.Parameter(torch.zeros(out_features, 1))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_left)
        nn.init.xavier_uniform_(self.a_right)

        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

        # Store the continuous adjacency values (with self-loops = 1.0)
        A = adjacency_matrix.clone()
        A = A + torch.eye(A.size(0))  # self-loops with weight 1
        self.register_buffer('adj_values', A)  # [N, N] — continuous weights
        # Binary mask for numerical stability: only mask true structural zeros
        self.register_buffer('adj_mask', (A.abs() > 0).float())  # [N, N]

    def forward(self, h):
        # h: [N, in_features]
        Wh = self.W(h)  # [N, out_features]

        # Compute attention logits: e_ij = LeakyReLU(a_left^T Wh_i + a_right^T Wh_j)
        e_left = Wh @ self.a_left    # [N, 1]
        e_right = Wh @ self.a_right  # [N, 1]
        e = e_left + e_right.T       # [N, N] via broadcast
        e = self.leaky_relu(e)

        # Mask only true structural zeros (where A_ij == 0) for softmax stability
        e = e.masked_fill(self.adj_mask == 0, float('-inf'))
        attention = F.softmax(e, dim=1)           # [N, N]

        # Valued masking: scale attention by continuous edge weights
        attention = attention * self.adj_values    # [N, N]

        attention = self.dropout(attention)

        return attention @ Wh  # [N, out_features]


class GAT(nn.Module):
    """Two-layer dense Graph Attention Network.

    Layer 1: multi-head attention (concatenation), ELU activation
    Layer 2: single-head attention, no activation (regression output)
    """
    def __init__(self, in_features, out_features, adjacency_matrix,
                 n_heads=4, dropout=0.6):
        super(GAT, self).__init__()
        assert in_features % n_heads == 0, (
            f"in_features ({in_features}) must be divisible by n_heads ({n_heads})")
        head_dim = in_features // n_heads

        # Layer 1: n_heads independent attention heads
        self.heads = nn.ModuleList([
            DenseGATLayer(in_features, head_dim, adjacency_matrix, dropout)
            for _ in range(n_heads)
        ])
        self.feat_dropout = nn.Dropout(dropout)

        # Layer 2: single-head attention producing the final output
        self.out_layer = DenseGATLayer(in_features, out_features,
                                       adjacency_matrix, dropout)

    def forward(self, x):
        # x: [N, in_features]
        x = self.feat_dropout(x)
        # Multi-head attention -> concatenate along feature dim
        x = torch.cat([head(x) for head in self.heads], dim=-1)  # [N, in_features]
        x = F.elu(x)
        x = self.feat_dropout(x)
        x = self.out_layer(x)  # [N, out_features]
        return x

    
def train_with_masking(model, data, optimizer, inductive=False):
        model.train()
        optimizer.zero_grad()
        if inductive:
            x_in = data.x[:, data.train_x_mask]
            out = model(x_in)
            y_target = data.y[:, data.train_y_mask].squeeze(-1)
            loss = F.l1_loss(out.squeeze(-1), y_target.float())
        else:
            out = model(data.x)[data.train_mask]
            loss = F.l1_loss(out.squeeze(-1), data.y[data.train_mask].float())
        loss.backward()
        optimizer.step()
        return loss

def test_with_masking(model, data, inductive=False):
    model.eval()
    with torch.no_grad():
        if inductive:
            x_in = data.x[:, data.train_x_mask]
            out = model(x_in)
            y_target = data.y[:, data.train_y_mask].squeeze(-1)
            train_loss = F.l1_loss(out.squeeze(-1), y_target.float()).item()
            
            x_in_val = data.x[:, data.val_x_mask]
            out_val = model(x_in_val)
            y_target_val = data.y[:, data.val_y_mask].squeeze(-1)
            val_loss = F.l1_loss(out_val.squeeze(-1), y_target_val.float()).item()
            
            x_in_test = data.x[:, data.test_x_mask]
            out_test = model(x_in_test)
            y_target_test = data.y[:, data.test_y_mask].squeeze(-1)
            test_loss = F.l1_loss(out_test.squeeze(-1), y_target_test.float()).item()
            
            return train_loss, val_loss, test_loss
        else:
            preds = model(data.x)
            # Evaluate on test nodes
            losses = []
            for mask in [data.train_mask, data.val_mask, data.test_mask]:
                pred = preds[mask].squeeze(-1)
                loss = F.l1_loss(pred, data.y[mask].float()).item()
                losses.append(loss)
            return losses

def rescaled_test_with_masking(model, data, node_scales, inductive=False, node_subset_mask=None):
    """Compute per-node rescaled MAE: mean(|pred_i - y_i| * scale_i).
    node_scales: 1-D tensor [N] with per-node (max - min) for undoing normalization.
    node_subset_mask: optional bool tensor [N]. If provided, losses are averaged
                      only over nodes where this mask is True.
    """
    model.eval()
    with torch.no_grad():
        if inductive:
            out = model(data.x[:, data.train_x_mask]).squeeze(-1)
            y = data.y[:, data.train_y_mask].squeeze(-1)
            errs = torch.abs(out - y) * node_scales
            train_loss = errs[node_subset_mask].mean().item() if node_subset_mask is not None else errs.mean().item()

            out_v = model(data.x[:, data.val_x_mask]).squeeze(-1)
            y_v = data.y[:, data.val_y_mask].squeeze(-1)
            errs_v = torch.abs(out_v - y_v) * node_scales
            val_loss = errs_v[node_subset_mask].mean().item() if node_subset_mask is not None else errs_v.mean().item()

            out_t = model(data.x[:, data.test_x_mask]).squeeze(-1)
            y_t = data.y[:, data.test_y_mask].squeeze(-1)
            errs_t = torch.abs(out_t - y_t) * node_scales
            test_loss = errs_t[node_subset_mask].mean().item() if node_subset_mask is not None else errs_t.mean().item()

            return train_loss, val_loss, test_loss
        else:
            preds = model(data.x).squeeze(-1)
            losses = []
            for mask in [data.train_mask, data.val_mask, data.test_mask]:
                if node_subset_mask is not None:
                    combined = mask & node_subset_mask
                else:
                    combined = mask
                loss = (torch.abs(preds[combined] - data.y[combined]) * node_scales[combined]).mean().item()
                losses.append(loss)
            return losses
    
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "raw/")
    connection_pd = pd.read_csv(os.path.join(data_dir, "line_station_connectionV1130.csv"))
    connection_pd
    
    # get station list from connection_pd (both station_cd1 and station_cd2) to ensure we have all stations
    station_list_connection_1 = connection_pd[['station_cd1', 'station_name1']].rename(columns={'station_cd1': 'station_id', 'station_name1': 'S12_001'}).drop_duplicates()
    station_list_connection_2 = connection_pd[['station_cd2', 'station_name2']].rename(columns={'station_cd2': 'station_id', 'station_name2': 'S12_001'}).drop_duplicates()
    station_list_pd = pd.concat([station_list_connection_1, station_list_connection_2]).drop_duplicates().reset_index(drop=True)

    # list of all the stations in the passenger survey data that are also in the connection data
    passenger_survey_pd = pd.read_csv(os.path.join(data_dir, "pass_survey_tokyov1109.csv"))
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
    #if there are multiple rows in the passenger survey for the same station_id, we take the mean of the passenger counts for that station_id across those rows, so that we have a single row per station_id that matches up with the nodes in the graph G:
    survey_agg = survey_filtered.groupby('station_id')[['2013','2014','2015','2016','2017','2018','2019']].mean()



    return {'G': G, 'survey_agg': survey_agg, 'station_node_pd': station_node_pd, 'matching_passenger_survey_pd': matching_passenger_survey_pd, 'node_order': node_order}

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # -------------------------- LOAD DATA ------------------
    path = os.path.join(os.path.dirname(__file__), "processed/data.pt")
    data = torch.load(path, weights_only=False)[0]
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    if aget(args, "inductive", False):
        split_path = os.path.join(root, "splits", "tokyo_railway_split_inductive.npz")

        with np.load(split_path) as f:
            train_x_mask = f["train_x_mask"]
            train_y_mask = f["train_y_mask"]
            val_x_mask   = f["val_x_mask"]
            val_y_mask   = f["val_y_mask"]
            test_x_mask  = f["test_x_mask"]
            test_y_mask  = f["test_y_mask"]

        data.train_x_mask = torch.tensor(train_x_mask, dtype=torch.bool)
        data.train_y_mask = torch.tensor(train_y_mask, dtype=torch.bool)
        data.val_x_mask = torch.tensor(val_x_mask, dtype=torch.bool)
        data.val_y_mask = torch.tensor(val_y_mask, dtype=torch.bool)
        data.test_x_mask = torch.tensor(test_x_mask, dtype=torch.bool)
        data.test_y_mask = torch.tensor(test_y_mask, dtype=torch.bool)
    else:
        # in the transductive case, our predictions must only be for the last year of data, i.e. the last column
        if data.y.dim() > 1:
            data.y = data.y[:, -1].squeeze(-1)  # make data.y 1D by taking only the last column and squeezing
        split_path = os.path.join(root, "splits", f"tokyo_railway_split_0.6_0.2_{args.seed}.npz")
        print(f"using split path with seed {args.seed}: {split_path}")

        with np.load(split_path) as f:
            train_mask = f["train_mask"]
            val_mask   = f["val_mask"]
            test_mask  = f["test_mask"]

        data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
        data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
        assert len(data.y[data.train_mask].shape) == 1, "Expected data.y[data.train_mask] to be 1D after squeezing"


    # ------------------------ LOAD DATA FOR COMPLEX ADJACENCY DETAILS ------------------

    data_dir = os.path.join(os.path.dirname(__file__), "raw/")
    connection_pd = pd.read_csv(os.path.join(data_dir, "line_station_connectionV1130.csv"))
    connection_pd
    
    # get station list from connection_pd (both station_cd1 and station_cd2) to ensure we have all stations
    station_list_connection_1 = connection_pd[['station_cd1', 'station_name1']].rename(columns={'station_cd1': 'station_id', 'station_name1': 'S12_001'}).drop_duplicates()
    station_list_connection_2 = connection_pd[['station_cd2', 'station_name2']].rename(columns={'station_cd2': 'station_id', 'station_name2': 'S12_001'}).drop_duplicates()
    station_list_pd = pd.concat([station_list_connection_1, station_list_connection_2]).drop_duplicates().reset_index(drop=True)

    # list of all the stations in the passenger survey data that are also in the connection data
    passenger_survey_pd = pd.read_csv(os.path.join(data_dir, "pass_survey_tokyov1109.csv"))
    passenger_survey_pd['station_id'] = passenger_survey_pd['station_id'].astype(int)
    passenger_survey_pd[passenger_survey_pd['station_id'].isin(station_list_pd['station_id'].values)]

    matching_passenger_survey_pd = passenger_survey_pd[passenger_survey_pd['station_id'].isin(station_list_pd['station_id'].values)].reset_index(drop=True)
    matching_passenger_survey_pd = matching_passenger_survey_pd.sort_values('station_id').reset_index(drop=True)

    # max-min normalize passenger counts:
    year_cols = ['2013', '2014', '2015', '2016', '2017', '2018', '2019']
    norm_mode = aget(args, "norm", "global")

    # Always compute global stats (needed to undo data.pt's baked-in global normalization)
    global_min = matching_passenger_survey_pd[year_cols].min().min()
    global_max = matching_passenger_survey_pd[year_cols].max().max()

    if norm_mode == "row":
        # Row (per-station) normalization for adjacency matrix computation
        row_mins = matching_passenger_survey_pd[year_cols].min(axis=1)
        row_maxs = matching_passenger_survey_pd[year_cols].max(axis=1)
        row_range = row_maxs - row_mins
        row_range = row_range.replace(0, 1)  # avoid div by zero for constant rows
        for year in year_cols:
            matching_passenger_survey_pd[year] = (matching_passenger_survey_pd[year] - row_mins) / row_range
    else:
        # Global normalization (default)
        for year in year_cols:
            matching_passenger_survey_pd[year] = (matching_passenger_survey_pd[year] - global_min) / (global_max - global_min)

    # Re-normalize data.x and data.y when using row normalization
    # (data.pt was saved with global normalization baked in)
    if norm_mode == "row":
        N_SURVEY = 6  # first 6 cols of data.x are survey years 2013-2018
        raw_survey_x = data.x[:, :N_SURVEY] * (global_max - global_min) + global_min
        if data.y.dim() == 1:
            raw_y = data.y * (global_max - global_min) + global_min
            all_years = torch.cat([raw_survey_x, raw_y.unsqueeze(1)], dim=1)
        else:
            raw_y = data.y * (global_max - global_min) + global_min
            all_years = torch.cat([raw_survey_x, raw_y], dim=1)
        node_row_mins = all_years.min(dim=1).values
        node_row_scales = all_years.max(dim=1).values - node_row_mins
        node_row_scales = torch.where(node_row_scales == 0, torch.ones_like(node_row_scales), node_row_scales)

        data.x[:, :N_SURVEY] = (raw_survey_x - node_row_mins.unsqueeze(1)) / node_row_scales.unsqueeze(1)
        if data.y.dim() == 1:
            data.y = (raw_y - node_row_mins) / node_row_scales
        else:
            data.y = (raw_y - node_row_mins.unsqueeze(1)) / node_row_scales.unsqueeze(1)
    else:
        node_row_scales = None

    # and now we need to make sure nodes in the passenger survey match up with nodes in the connection data
    # the passenger_survey_pd only has nodes that are in the station_list_pd, so we can just filter the station_list_pd to only include those nodes
    # note that both data had stations that were not in the other
    station_list_pd = station_list_pd[station_list_pd['station_id'].isin(matching_passenger_survey_pd['station_id'].values)].reset_index(drop=True)
    station_list_pd = station_list_pd.sort_values('station_id').reset_index(drop=True)

    station_node_pd = connection_pd[connection_pd['station_cd1'].isin(station_list_pd['station_id'].values) & connection_pd['station_cd2'].isin(station_list_pd['station_id'].values)][['line', 'station_cd1', 'station_cd2', 'distance']].drop_duplicates()

    # build graph structure using networkx
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
    #if there are multiple rows in the passenger survey for the same station_id, we take the mean of the passenger counts for that station_id across those rows, so that we have a single row per station_id that matches up with the nodes in the graph G:
    survey_agg = survey_filtered.groupby('station_id')[['2013','2014','2015','2016','2017','2018','2019']].mean()

    #incorporate lat/long information to survey_agg for distance-based adjacency matrix:
    node_ordered_survey_agg = survey_agg.loc[node_order].copy()
    node_ordered_survey_agg['lat'] = float('nan')
    node_ordered_survey_agg['lng'] = float('nan')

    for station_id in node_order:
        row1 = connection_pd[connection_pd['station_cd1'] == station_id][['station_lat1', 'station_lng1']].head(1)
        if not row1.empty:
            node_ordered_survey_agg.loc[station_id, 'lat'] = row1['station_lat1'].values[0]
            node_ordered_survey_agg.loc[station_id, 'lng'] = row1['station_lng1'].values[0]
        else:
            row2 = connection_pd[connection_pd['station_cd2'] == station_id][['station_lat2', 'station_lng2']].head(1)
            if not row2.empty:
                node_ordered_survey_agg.loc[station_id, 'lat'] = row2['station_lat2'].values[0]
                node_ordered_survey_agg.loc[station_id, 'lng'] = row2['station_lng2'].values[0]

    # ------------------------- CALCULATE PAIRWISE DISTANCES FOR DISTANCE-BASED ADJACENCY MATRIX ------------------
    def get_distance_adjacency_matrix():
        #calculate all the haversine distances to create the distance adjacency matrix (vectorized)
        R = 3958.8  # Earth radius in miles

        lats = np.radians(node_ordered_survey_agg['lat'].values)   # shape [N]
        lngs = np.radians(node_ordered_survey_agg['lng'].values)   # shape [N]

        # Broadcast to [N, N]
        dphi    = lats[:, None] - lats[None, :]
        dlambda = lngs[:, None] - lngs[None, :]

        a = np.sin(dphi / 2)**2 + np.cos(lats[:, None]) * np.cos(lats[None, :]) * np.sin(dlambda / 2)**2
        dist_np = 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        # we have to invert the distances to get a similarity measure, so closer stations are more "correlated"
        reverse_scaled_dist = np.exp(-dist_np/(args.sigma ** 2))

        distance_adjacency_matrix = torch.tensor(reverse_scaled_dist, dtype=torch.float32)
        return distance_adjacency_matrix

    # ----------------------- CALCULATE CORRELATION-BASED ADJACENCY MATRIX ----------------------
    # calculate pairwise Pearson correlation between rows of node_ordered_survey_agg[year_cols], which will be our correlation-based adjacency matrix:
    # faster way to compute correlation adjacency matrix using numpy broadcasting (but less memory efficient)
    def get_correlation_adjacency_matrix():
        data_matrix = node_ordered_survey_agg[year_cols].values  # shape [N, T]
        data_matrix_centered = data_matrix - data_matrix.mean(axis=1, keepdims=True)  # center each node's time series
        covariance_matrix = np.dot(data_matrix_centered, data_matrix_centered.T) / (data_matrix.shape[1] - 1)  # shape [N, N]
        std_dev = np.sqrt(np.diag(covariance_matrix))  # shape [N]
        correlation_adjacency_matrix = covariance_matrix / np.outer(std_dev, std_dev)  # shape [N, N]
        correlation_adjacency_matrix = np.nan_to_num(correlation_adjacency_matrix)  # replace NaNs with 0 (in case of constant rows)
        correlation_adjacency_matrix = torch.tensor(correlation_adjacency_matrix, dtype=torch.float32)
        return correlation_adjacency_matrix.fill_diagonal_(0)  # zero out diagonal to remove self-loops in correlation adjacency
    
    # ------------------------- TRAINING ------------------
    default_adjacency_matrix = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
    adjacency_type = args.adjacency
    if adjacency_type == "con":
        adjacency_matrix = default_adjacency_matrix
    elif adjacency_type == "dis":
        adjacency_matrix = get_distance_adjacency_matrix()
    elif adjacency_type == "cor":
        adjacency_matrix = get_correlation_adjacency_matrix()
    elif adjacency_type == "d_con":
        adjacency_matrix = torch.mul(get_distance_adjacency_matrix(), default_adjacency_matrix)
    elif adjacency_type == "d_cor":
        adjacency_matrix = torch.mul(get_distance_adjacency_matrix(), get_correlation_adjacency_matrix())
    elif adjacency_type == "cor_con":
        adjacency_matrix = torch.mul(get_correlation_adjacency_matrix(), default_adjacency_matrix)
    elif adjacency_type == "d_cor_con":
        adjacency_matrix = torch.mul(get_distance_adjacency_matrix(), get_correlation_adjacency_matrix())
        adjacency_matrix = torch.mul(adjacency_matrix, default_adjacency_matrix)
    else:
        raise ValueError(f"Unknown adjacency type: {adjacency_type}")
    
    inductive = aget(args, "inductive", False)
    in_features = data.train_x_mask.sum().item() if inductive else data.x.shape[1]
    model_type = aget(args, "model", "gcn")
    if model_type == "gat":
        n_heads = aget(args, "n_heads", 4)
        dropout = aget(args, "dropout", 0.6)
        model = GAT(in_features=in_features, out_features=1,
                    adjacency_matrix=adjacency_matrix,
                    n_heads=n_heads, dropout=dropout)
    else:
        model = GCN(in_features=in_features, out_features=1, adjacency_matrix=adjacency_matrix)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    t_train_start = time.perf_counter()
    for epoch in range(500):
        loss = train_with_masking(model, data, optimizer, inductive=inductive)
        if (epoch + 1) % 20 == 0:
            train_loss, val_loss, test_loss = test_with_masking(model, data, inductive=inductive)
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.10f}, Val Loss: {val_loss:.10f}, Test Loss: {test_loss:.10f}")
    train_time_s = time.perf_counter() - t_train_start
    print(f"Training wall-clock time: {train_time_s:.2f}s")
    
    # rescale losses back to original scale (undo normalization) for interpretability:
    train_loss, val_loss, test_loss = test_with_masking(model, data, inductive=inductive)
    if norm_mode == "row":
        average_p_counts = node_ordered_survey_agg[year_cols].mean(axis=1) * node_row_scales.numpy() + node_row_mins.numpy()
        average_p_counts = average_p_counts.values  # convert from pandas Series to numpy array
        subset_masks = build_node_subset_masks(average_p_counts)

        # Compute rescaled MAE for every subset automatically
        subset_results = {}
        for subset_name, smask in subset_masks.items():
            s_train, s_val, s_test = rescaled_test_with_masking(
                model, data, node_row_scales, inductive=inductive, node_subset_mask=smask)
            subset_results[subset_name] = (s_train, s_val, s_test)
            print(f"  [{subset_name}] Train MAE (rescaled): {s_train:.4f}, Val: {s_val:.4f}, Test: {s_test:.4f}")

        # "all" is the default headline metric
        original_train_loss, original_val_loss, original_test_loss = subset_results["all"]
    else:
        original_train_loss = train_loss * (global_max - global_min)
        original_val_loss = val_loss * (global_max - global_min)
        original_test_loss = test_loss * (global_max - global_min)
        subset_results = None
    print(f"Final Train Loss (rescaled): {original_train_loss:.10f}, Val Loss (rescaled): {original_val_loss:.10f}, Test Loss (rescaled): {original_test_loss:.10f}")
    return original_train_loss, original_val_loss, original_test_loss, train_loss, val_loss, test_loss, train_time_s, subset_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adjacency", type=str, default="con", help="Type of adjacency matrix to use")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--inductive", action="store_true", help="Whether to use inductive learning setting (with column masking) instead of transductive")
    parser.add_argument("--sigma", type=float, default=5.0, help="Sigma smoothing parameter for distance-based adjacency matrix")
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gat"], help="Model architecture to use")
    parser.add_argument("--n_heads", type=int, default=2, help="Number of attention heads for GAT")
    parser.add_argument("--dropout", type=float, default=0.6, help="Dropout rate for GAT")
    parser.add_argument("--norm", type=str, default="global", choices=["global", "row"],
                        help="Normalization mode: 'global' (single min/max across all stations) or 'row' (per-station min/max)")
    parser.add_argument("--sweep", action="store_true", help="Run all 7 adjacency types and write results to CSV, with an average over n random seeds")
    parser.add_argument("--sweep_seeds", type=int, default=3, help="Number of random seeds to use for sweep (only if --sweep is set)")
    args = parser.parse_args()

    if args.sweep:
        import csv
        adjacency_types = ["con", "dis", "cor", "d_con", "d_cor", "cor_con", "d_cor_con"]
        seeds = list(range(args.sweep_seeds))
        mode = "inductive" if args.inductive else "transductive"
        out_path = os.path.join(os.path.dirname(__file__), f"{args.model}_sweep_{mode}_norm{args.norm}_seed{args.seed}_sigma{args.sigma}.csv")
        rows = []
        for adj_type in adjacency_types:
            train_losses = []
            val_losses = []
            test_losses = []
            normalized_train_losses = []
            normalized_val_losses = []
            normalized_test_losses = []
            run_times = []
            for seed in seeds:
                args.seed = seed
                print(f"\n{'='*60}")
                print(f"Running adjacency type: {adj_type}, seed: {seed}")
                print(f"{'='*60}")
                args.adjacency = adj_type
                train_loss, val_loss, test_loss, normalized_train_loss, normalized_val_loss, normalized_test_loss, train_time_s, subset_results = main(args)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                test_losses.append(test_loss)
                normalized_train_losses.append(normalized_train_loss)
                normalized_val_losses.append(normalized_val_loss)
                normalized_test_losses.append(normalized_test_loss)
                run_times.append(train_time_s)
                top_10_test_mae = subset_results["top10"][2] if subset_results is not None else None
                top_100_test_mae = subset_results["top100"][2] if subset_results is not None else None
                bottom_100_test_mae = subset_results["bottom100"][2] if subset_results is not None else None
            rows.append({
                "adjacency": adj_type,
                "mode": mode,
                "norm": args.norm,
                "seed": args.seed,
                "sigma": args.sigma,
                "train_mae": f"{np.mean(train_losses):.4f}",
                "val_mae": f"{np.mean(val_losses):.4f}",
                "test_mae": f"{np.mean(test_losses):.4f}",
                "normalized_train_mae": f"{np.mean(normalized_train_losses):.4f}",
                "normalized_val_mae": f"{np.mean(normalized_val_losses):.4f}",
                "normalized_test_mae": f"{np.mean(normalized_test_losses):.4f}",
                "train_mae_std": f"{np.std(train_losses):.4f}",
                "val_mae_std": f"{np.std(val_losses):.4f}",
                "test_mae_std": f"{np.std(test_losses):.4f}",
                "normalized_train_mae_std": f"{np.std(normalized_train_losses):.4f}",
                "normalized_val_mae_std": f"{np.std(normalized_val_losses):.4f}",
                "normalized_test_mae_std": f"{np.std(normalized_test_losses):.4f}",
                "avg_train_time_s": f"{np.mean(run_times):.2f}",
                "std_train_time_s": f"{np.std(run_times):.2f}",
                "top_10_test_mae": f"{top_10_test_mae:.4f}" if top_10_test_mae is not None else "None",
                "top_100_test_mae": f"{top_100_test_mae:.4f}" if top_100_test_mae is not None else "None",
                "bottom_100_test_mae": f"{bottom_100_test_mae:.4f}" if bottom_100_test_mae is not None else "None"
            })
            print(f"  {adj_type} avg training time: {np.mean(run_times):.2f}s +/- {np.std(run_times):.2f}s")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults written to {out_path}")
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        main(args)