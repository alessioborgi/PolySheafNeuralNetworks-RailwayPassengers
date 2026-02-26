import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import numpy as np
import pandas as pd
import networkx as nx

import random

import argparse
import os

from exp.run import aget

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
        x = self.layer_2(x)
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
        split_path = os.path.join(root, "splits", "tokyo_railway_split_0.6_0.2_0.npz")

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

    # ----------------------- CALCULATE CORRELATION-BASED ADJACENCY MATRIX ----------------------
    # calculate pairwise Pearson correlation between rows of node_ordered_survey_agg[year_cols], which will be our correlation-based adjacency matrix:
    # faster way to compute correlation adjacency matrix using numpy broadcasting (but less memory efficient)
    data_matrix = node_ordered_survey_agg[year_cols].values  # shape [N, T]
    # 
    data_matrix_centered = data_matrix - data_matrix.mean(axis=1, keepdims=True)  # center each node's time series
    covariance_matrix = np.dot(data_matrix_centered, data_matrix_centered.T) / (data_matrix.shape[1] - 1)  # shape [N, N]
    std_dev = np.sqrt(np.diag(covariance_matrix))  # shape [N]
    correlation_adjacency_matrix = covariance_matrix / np.outer(std_dev, std_dev)  # shape [N, N]
    correlation_adjacency_matrix = np.nan_to_num(correlation_adjacency_matrix)  # replace NaNs with 0 (in case of constant rows)
    correlation_adjacency_matrix = torch.tensor(correlation_adjacency_matrix, dtype=torch.float32)
    
    # ------------------------- TRAINING ------------------
    default_adjacency_matrix = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
    adjacency_type = args.adjacency
    if adjacency_type == "con":
        adjacency_matrix = default_adjacency_matrix
    elif adjacency_type == "dis":
        adjacency_matrix = distance_adjacency_matrix
    elif adjacency_type == "cor":
        adjacency_matrix = correlation_adjacency_matrix.fill_diagonal_(0)  # zero out diagonal to remove self-loops in correlation adjacency
    elif adjacency_type == "d_con":
        adjacency_matrix = torch.mul(distance_adjacency_matrix, default_adjacency_matrix)
    elif adjacency_type == "d_cor":
        adjacency_matrix = torch.mul(distance_adjacency_matrix, correlation_adjacency_matrix)
    elif adjacency_type == "cor_con":
        adjacency_matrix = torch.mul(correlation_adjacency_matrix, default_adjacency_matrix)
    elif adjacency_type == "d_cor_con":
        adjacency_matrix = torch.mul(distance_adjacency_matrix, correlation_adjacency_matrix)
        adjacency_matrix = torch.mul(adjacency_matrix, default_adjacency_matrix)
    else:
        raise ValueError(f"Unknown adjacency type: {adjacency_type}")
    
    inductive = aget(args, "inductive", False)
    in_features = data.train_x_mask.sum().item() if inductive else data.x.shape[1]
    model = GCN(in_features=in_features, out_features=1, adjacency_matrix=adjacency_matrix)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(500):
        loss = train_with_masking(model, data, optimizer, inductive=inductive)
        if (epoch + 1) % 20 == 0:
            train_loss, val_loss, test_loss = test_with_masking(model, data, inductive=inductive)
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.10f}, Val Loss: {val_loss:.10f}, Test Loss: {test_loss:.10f}")
    
    # rescale losses back to original scale (undo normalization) for interpretability:
    train_loss, val_loss, test_loss = test_with_masking(model, data, inductive=inductive)
    original_train_loss = train_loss * (global_max - global_min) #+ global_min
    original_val_loss = val_loss * (global_max - global_min) #+ global_min
    original_test_loss = test_loss * (global_max - global_min) #+ global_min
    print(f"Final Train Loss (rescaled): {original_train_loss:.10f}, Val Loss (rescaled): {original_val_loss:.10f}, Test Loss (rescaled): {original_test_loss:.10f}")
    return original_train_loss, original_val_loss, original_test_loss, train_loss, val_loss, test_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adjacency", type=str, default="con", help="Type of adjacency matrix to use")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--inductive", action="store_true", help="Whether to use inductive learning setting (with column masking) instead of transductive")
    parser.add_argument("--sigma", type=float, default=5.0, help="Sigma smoothing parameter for distance-based adjacency matrix")
    parser.add_argument("--sweep", action="store_true", help="Run all 7 adjacency types and write results to CSV")
    args = parser.parse_args()

    if args.sweep:
        import csv
        adjacency_types = ["con", "dis", "cor", "d_con", "d_cor", "cor_con", "d_cor_con"]
        mode = "inductive" if args.inductive else "transductive"
        out_path = os.path.join(os.path.dirname(__file__), f"gcn_sweep_{mode}_seed{args.seed}_sigma{args.sigma}.csv")
        rows = []
        for adj_type in adjacency_types:
            print(f"\n{'='*60}")
            print(f"Running adjacency type: {adj_type}")
            print(f"{'='*60}")
            args.adjacency = adj_type
            train_loss, val_loss, test_loss, normalized_train_loss, normalized_val_loss, normalized_test_loss = main(args)
            rows.append({
                "adjacency": adj_type,
                "mode": mode,
                "seed": args.seed,
                "sigma": args.sigma,
                "train_mae": f"{train_loss:.4f}",
                "val_mae": f"{val_loss:.4f}",
                "test_mae": f"{test_loss:.4f}",
                "normalized_train_mae": f"{normalized_train_loss:.4f}",
                "normalized_val_mae": f"{normalized_val_loss:.4f}",
                "normalized_test_mae": f"{normalized_test_loss:.4f}",
            })
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults written to {out_path}")
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        main(args)