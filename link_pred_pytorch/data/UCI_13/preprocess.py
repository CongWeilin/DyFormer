import dill
from collections import defaultdict
from datetime import datetime, timedelta

from scipy.sparse import csr_matrix

import networkx as nx
import numpy as np

links = []
ts = []
ctr = 0
node_cnt = 0
node_idx = {}
idx_node = []

with open('./out.opsahl-ucsocial') as f:
    lines = f.read().splitlines()
    for l in lines:
        if l[0] == '%':
            continue
            
        x, y, e, t = map(int, l.split())
        # print (x,y,e,t)
        timestamp = datetime.fromtimestamp(t)
        ts.append(timestamp)
        
        ctr += 1
        if ctr % 100000 == 0:
            print (ctr)
            
        if x not in node_idx:
            node_idx[x] = node_cnt 
            node_cnt += 1
            
        if y not in node_idx:
            node_idx[y] = node_cnt 
            node_cnt += 1
    
        links.append((node_idx[x],node_idx[y], timestamp))

print ("Min ts", min(ts), "max ts", max(ts))    
print ("Total time span: {} days".format((max(ts) - min(ts)).days))
links.sort(key =lambda x: x[2])


SLICE_DAYS = 10
START_DATE = min(ts) + timedelta(5)
END_DATE = max(ts) - timedelta(60)

slices_links = defaultdict(lambda : nx.MultiGraph())
slices_features = defaultdict(lambda : {})

print ("Start date", START_DATE)

slice_id = 0
# Split the set of links in order by slices to create the graphs. 
for (a, b, time) in links:
    prev_slice_id = slice_id
    
    datetime_object = time
    if datetime_object < START_DATE:
        continue
    if datetime_object > END_DATE:
        break
        days_diff = (END_DATE - START_DATE).days
    else:
        days_diff = (datetime_object - START_DATE).days
        
    
    slice_id = days_diff // SLICE_DAYS
    
    if slice_id == 1+prev_slice_id and slice_id > 0:
        slices_links[slice_id] = nx.MultiGraph()
        slices_links[slice_id].add_nodes_from(slices_links[slice_id-1].nodes(data=True))
        assert (len(slices_links[slice_id].edges()) ==0)
        #assert len(slices_links[slice_id].nodes()) >0

    if slice_id == 1+prev_slice_id and slice_id ==0:
        slices_links[slice_id] = nx.MultiGraph()

    if a not in slices_links[slice_id]:
        slices_links[slice_id].add_node(a)
    if b not in slices_links[slice_id]:
        slices_links[slice_id].add_node(b)    
    slices_links[slice_id].add_edge(a,b, date=datetime_object)
    
for slice_id in slices_links:
    print ("# nodes in slice", slice_id, len(slices_links[slice_id].nodes()))
    print ("# edges in slice", slice_id, len(slices_links[slice_id].edges()))
    
    temp = np.identity(len(slices_links[max(slices_links.keys())].nodes()))
    print ("Shape of temp matrix", temp.shape)
    slices_features[slice_id] = {}
    for idx, node in enumerate(slices_links[slice_id].nodes()):
        slices_features[slice_id][node] = temp[idx]


def remap(slices_graph, slices_features):
    all_nodes = []
    for slice_id in slices_graph:
        assert len(slices_graph[slice_id].nodes()) == len(slices_features[slice_id])
        all_nodes.extend(slices_graph[slice_id].nodes())
    all_nodes = list(set(all_nodes))
    print ("Total # nodes", len(all_nodes), "max idx", max(all_nodes))
    ctr = 0
    node_idx = {}
    idx_node = []
    for slice_id in slices_graph:
        for node in slices_graph[slice_id].nodes():
            if node not in node_idx:
                node_idx[node] = ctr
                idx_node.append(node)
                ctr += 1
    slices_graph_remap = []
    slices_features_remap = []
    for slice_id in slices_graph:
        G = nx.MultiGraph()
        for x in slices_graph[slice_id].nodes():
            G.add_node(node_idx[x])
        for x in slices_graph[slice_id].edges(data=True):
            G.add_edge(node_idx[x[0]], node_idx[x[1]], date=x[2]['date'])
        assert (len(G.nodes()) == len(slices_graph[slice_id].nodes()))
        assert (len(G.edges()) == len(slices_graph[slice_id].edges()))
        slices_graph_remap.append(G)
    
    for slice_id in slices_features:
        features_remap = []
        for x in slices_graph_remap[slice_id].nodes():
            features_remap.append(slices_features[slice_id][idx_node[x]])
            #features_remap.append(np.array(slices_features[slice_id][idx_node[x]]).flatten())
        features_remap = csr_matrix(np.squeeze(np.array(features_remap)))
        slices_features_remap.append(features_remap)
    return (slices_graph_remap, slices_features_remap)

slices_links_remap, slices_features_remap = remap(slices_links, slices_features)

np.savez('graphs.npz', graph=slices_links_remap)
np.savez('features.npz', feats=slices_features_remap)

