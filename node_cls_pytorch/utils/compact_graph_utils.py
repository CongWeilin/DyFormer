import networkx as nx
import numpy as np
from collections import defaultdict

###############################################################################
###############################################################################
###############################################################################

def get_k_hop_neighbors(graph, k_hop):
    shortest_path = dict(nx.shortest_path_length(graph, weight='compact_G_weight'))
    
    k_hop_neighbors = defaultdict(lambda: [])

    for src_node in graph.nodes:
        pair_wise_dist = shortest_path[src_node]
        for dst_node, distance in pair_wise_dist.items():
            if distance <= k_hop:
                k_hop_neighbors[src_node].append(dst_node)
                
    return k_hop_neighbors, shortest_path

###############################################################################
###############################################################################
###############################################################################

def create_compact_graph(graphs):
    compact_G = nx.DiGraph()
    compact_G_node_list = []
    
    # compress multi-graphs to one compact graph
    num_graphs = len(graphs)
    num_nodes_each_graph = [0]
    
    for i in range(num_graphs):
        num_nodes_each_graph.append(num_nodes_each_graph[i] + len(graphs[i].nodes))
        
        for node in graphs[i].nodes:
            compact_G.add_node(node + num_nodes_each_graph[i])
            compact_G_node_list.append(node)
            
        for edge in graphs[i].edges:
            if edge[0] >= edge[1]:
                continue
            compact_G.add_edge(edge[0] + num_nodes_each_graph[i], 
                               edge[1] + num_nodes_each_graph[i], edge_type=0, compact_G_weight=1)
            compact_G.add_edge(edge[1] + num_nodes_each_graph[i], 
                               edge[0] + num_nodes_each_graph[i], edge_type=0, compact_G_weight=1)
            
    # add temporal links
    for i in range(num_graphs-1):
        node_list_a = list(graphs[i].nodes)
        node_list_b = list(graphs[i+1].nodes)
        node_list_intersect = np.intersect1d(node_list_a, node_list_b)
        
        for node in node_list_intersect:
            compact_G.add_edge(node + num_nodes_each_graph[i+1], 
                               node + num_nodes_each_graph[i], edge_type=1, compact_G_weight=1)
    
    compact_adj = nx.adjacency_matrix(compact_G)
    return compact_G, compact_adj, num_nodes_each_graph

###############################################################################
###############################################################################
###############################################################################


def get_graph_dst_src_time_dist_info(graphs, k_hop=3):

    compact_G, compact_adj, num_nodes_each_graph = create_compact_graph(graphs)

    compact_id_to_ori = np.zeros(num_nodes_each_graph[-1])
    for i in range(len(num_nodes_each_graph)-1):
        compact_id_to_ori[num_nodes_each_graph[i]: num_nodes_each_graph[i+1]] = i

    k_hop_neighbors, shortest_path = get_k_hop_neighbors(compact_G, k_hop)
    # for each node in the last training graph
    graph_dst_src_time_dist_list = []

    for train_end in range(len(graphs)):
        ###        
        dst_id  = []
        src_id  = []
        time_id = []
        dist_id = [] 
            
        for node in graphs[train_end].nodes:
            compact_node_id = node + num_nodes_each_graph[train_end]
            neighbors_ = k_hop_neighbors[compact_node_id]

            # add other edges
            for i in range(len(neighbors_)):
                if shortest_path[compact_node_id][neighbors_[i]] == 0:
                    continue
                    
                dst_id.append(compact_node_id)
                src_id.append(neighbors_[i])
                
                time_id_ = train_end - compact_id_to_ori[neighbors_[i]]
                dist_id_ = shortest_path[compact_node_id][neighbors_[i]] - time_id_
                time_id.append(time_id_)
                dist_id.append(dist_id_)

        
        ###
        visible_nodes = np.unique([src_id + dst_id])
        # add self-loop 
        for node in visible_nodes:
            dst_id.append(node)
            src_id.append(node)
            time_id.append(0)
            dist_id.append(0)
        
        dst_src_time_dist = np.array([dst_id, src_id, time_id, dist_id], dtype=np.int32)
        graph_dst_src_time_dist_list.append(dst_src_time_dist)
    
    return graph_dst_src_time_dist_list, num_nodes_each_graph

###############################################################################
###############################################################################
###############################################################################

# contrastive pairs

def get_intersect_nodes(graphs):
    num_graphs = len(graphs)

    nodes = []
    for G in graphs:
        nodes.append(np.unique(G.edges))

    intersect_nodes_1hop = []
    for i in range(num_graphs-1):
        intersect = np.intersect1d(nodes[i], nodes[i+1])
        intersect_nodes_1hop.append(intersect)

    intersect_nodes_2hop = []
    for i in range(num_graphs-2):
        intersect = np.intersect1d(nodes[i], nodes[i+2])
        intersect_nodes_2hop.append(intersect)

    intersect_nodes_3hop = []
    for i in range(num_graphs-3):
        intersect = np.intersect1d(nodes[i], nodes[i+3])
        intersect_nodes_3hop.append(intersect)
        
    return intersect_nodes_1hop, intersect_nodes_2hop, intersect_nodes_3hop