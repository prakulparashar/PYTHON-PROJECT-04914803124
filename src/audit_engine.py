import osmnx as ox
import networkx as nx

def get_city_network(place_name):
    # Download and project the graph
    G = ox.graph_from_place(place_name, network_type='walk')
    G = ox.project_graph(G)
    
    # Add travel time to edges
    speed = 4.5 # km/h
    meters_per_min = (speed * 1000) / 60
    for u, v, k, data in G.edges(data=True, keys=True):
        data['time'] = data['length'] / meters_per_min
    return G

def get_pois(place_name, category):
    # Define tags based on the amenity category
    tags = {'amenity': category} if category != 'supermarket' else {'shop': 'supermarket'}
    pois = ox.features_from_place(place_name, tags)
    return pois[pois.geometry.type == 'Point']