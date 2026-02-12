import csv
import torch
import pandas as pd

def load_tokyo_railway_passengers():
    """
    Loads the Tokyo railway passengers dataset.

    Returns:
        A list of tuples, where each tuple contains the station, year, and the number of passengers.
    """
    data = {}
    with open('datasets/tokyo_railway/graph_passenger_survey202411.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            station = row[0]
            year = row[1]
            passengers = int(row[2])
            # line = row[3]

            data.append((station, None, None, None))
    return data

if __name__ == "__main__":
    data = load_tokyo_railway_passengers()
    # for i, entry in enumerate(set(data)):  # Print the unique entries
    #     if i >= 10:  # Limit to the first 10 unique entries for brevity
    #         break
    #     print(entry)
    print("length of dataset:", len(set(data)))

    # Let's build a graph representation of the Tokyo railway passengers dataset. We will create a graph where each station is a node, and edges represent the connections between stations. The number of passengers can be used as edge weights.
    import networkx as nx

    def build_graph(data):
        G = nx.Graph()
        for station, year, passengers, line in data:
            G.add_node(station)
            G.adjacency
            # Here you would add edges based on your specific criteria
            # For example, if you have a list of connections between stations:
            # for connection in connections:
            #     G.add_edge(connection[0], connection[1], weight=connection[2])
        return G
    graph = build_graph(data)
    print(graph.nodes["Ikebukuro"])  # Example of accessing a node

    def build_connection_adjacency_matrix():
        data = load_tokyo_railway_passengers()
        stations = set([entry[0] for entry in data])
        station_to_index = {station: idx for idx, station in enumerate(stations)}
        index_to_station = {idx: station for station, idx in station_to_index.items()}
        adjacency_matrix = torch.zeros((len(stations), len(stations)))
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                pass


