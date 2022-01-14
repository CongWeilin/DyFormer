import networkx as nx
import torch
import numpy as np

from .random_walk import Graph_RandomWalk

#########################################################################
#########################################################################
#########################################################################
def create_compact_graph(graphs):
    compact_G = nx.DiGraph()
    compact_G_node_list = []
    
    # compress multi-graphs to one compact graph
    num_graphs = len(graphs)
    num_nodes_each_graph = [0]
    
    # add spatial edges
    for i in range(num_graphs):
        num_nodes_each_graph.append(num_nodes_each_graph[i] + len(graphs[i].nodes))
        
        for node in graphs[i].nodes:
            compact_G.add_node(node + num_nodes_each_graph[i])
            compact_G_node_list.append(node)
            
        for edge in graphs[i].edges:
            if edge[0] >= edge[1]:
                continue
            compact_G.add_edge(edge[0] + num_nodes_each_graph[i], 
                               edge[1] + num_nodes_each_graph[i], edge_type=0)
            compact_G.add_edge(edge[1] + num_nodes_each_graph[i], 
                               edge[0] + num_nodes_each_graph[i], edge_type=0)

    # add temporal links
    for i in range(num_graphs-1):
        node_list_a = list(graphs[i].nodes)
        node_list_b = list(graphs[i+1].nodes)
        node_list_intersect = np.intersect1d(node_list_a, node_list_b)
        
        for node in node_list_intersect:
            compact_G.add_edge(node + num_nodes_each_graph[i], 
                               node + num_nodes_each_graph[i+1], edge_type=1)
    
    for edge in compact_G.edges():
        compact_G[edge[0]][edge[1]]['weight'] = 1

    return compact_G, num_nodes_each_graph

#########################################################################
#########################################################################
#########################################################################
def get_compact_adj_edges(compact_graph_train):

    all_edges_indices = []
    all_edges_types   = []
    for edge in compact_graph_train.edges:
        edge_type = compact_graph_train.edges[edge]['edge_type']
        all_edges_indices.append([edge[0], edge[1]])
        all_edges_types.append(edge_type)
        
    return all_edges_indices, all_edges_types

#########################################################################
#########################################################################
#########################################################################


def align_output(num_nodes_graph_eval, window, num_nodes_each_graph, output):
    output_aligned = torch.zeros(num_nodes_graph_eval, window, output.size(-1)).to(output)

    for i in range(window):
        start_id = num_nodes_each_graph[i]
        end_id   = num_nodes_each_graph[i+1]
        output_aligned[:end_id-start_id, i, :] = output[start_id : end_id, :]
        
    return output_aligned

#########################################################################
#########################################################################
#########################################################################

def get_randomwalk_neighbors(compact_graph_train, num_nodes_each_graph):
    
    # sample neighbors with random walks
    graph_randomwalk = Graph_RandomWalk(compact_graph_train, True, 1.0, 1.0)
    graph_randomwalk.preprocess_transition_probs()
    all_nodes = graph_randomwalk.G.nodes

    edges_list_0 = []
    edges_list_1 = []
    edges_val_list = []
    for node in all_nodes:
        walks = []
        for _ in range(100):
            walk = graph_randomwalk.node2vec_walk(walk_length=3, start_node=node)
            walks.extend(walk)

        unique, counts = np.unique(walks, return_counts=True)
        prob = counts/np.sum(counts)

        for i in range(len(unique)):
            if isinstance(unique[i], list):
                print(node, unique[i], prob[i], walks)
            edges_list_0.append(node)
            edges_list_1.append(unique[i])
            edges_val_list.append(prob[i])
    
    # compute node timestep    
    node_time_idx = []
    for t in range(len(num_nodes_each_graph)-1):
        node_time_idx.append(t * np.ones(num_nodes_each_graph[t+1]-num_nodes_each_graph[t]))
    node_time_idx = np.concatenate(node_time_idx).astype(np.int32)

    # get the edge type
    edges_type_list = []
    for e0, e1 in zip(edges_list_0, edges_list_1):
        if e0 == e1:
            edges_type_list.append(2) # self-loop edge type
        elif node_time_idx[e0] == node_time_idx[e1]:
            edges_type_list.append(1) # same timestep
        elif node_time_idx[e0] != node_time_idx[e1]:
            edges_type_list.append(0)
            
    edges_list = np.transpose(np.array([edges_list_0, edges_list_1], dtype=np.int32))
    edges_val_list = np.array(edges_val_list, dtype=np.float32)
    edges_type_list = np.array(edges_type_list, dtype=np.int32)
    
    return edges_list, edges_val_list, edges_type_list
