import dill
from collections import defaultdict
from datetime import datetime, timedelta
from scipy.sparse import csr_matrix
import networkx as nx
import numpy as np

# extract nodes for node index preparation
users = []
items = []
with open('./reddit.csv') as f:
    next(f)
    lines = f.read().splitlines()
    for l in lines:
        if l[0] == '%':
            continue
        e = l.strip().split(',')
        user_id = int(e[0])
        item_id = int(e[1])

        timestamp = float(e[2])
        state_label = float(e[3])
        feature = np.array([float(x) for x in e[4:]])

        users.append(user_id)
        items.append(item_id)

user_set = list(set(users))

# prepare node dictionary: key to node index
node_cnt = 0
node_idx = {}

for u in users:
    if "u_"+str(u) not in node_idx:
        node_idx["u_"+str(u)] = node_cnt
        node_cnt += 1

print('##############################')
print('Max user id: ', node_cnt-1)
print('##############################')

for i in items: 
    if "i_"+str(i) not in node_idx:
        node_idx["i_"+str(i)] = node_cnt
        node_cnt += 1
print ("# nodes", node_cnt)

# extract edges: links list stores everything
ctr = 0
links = []
time_list = []
with open('./reddit.csv') as f:
    next(f)
    lines = f.read().splitlines()
    for l in lines:
        if l[0] == '%':
            continue
        e = l.strip().split(',')
        user_id = int(e[0])
        item_id = int(e[1])

        timestamp = float(e[2])
        state_label = float(e[3])
        feature = np.array([float(x) for x in e[4:]])
        
        ctr += 1
        if ctr % 10000 == 0:
            print(ctr)
        
        time_list.append(timestamp)
        links.append((node_idx["u_"+str(user_id)],node_idx["i_"+str(item_id)], timestamp, state_label, feature))

# stdout some statistics
print ("Min ts", min(time_list), "max ts", max(time_list))
print ("Total time span: {}".format(max(time_list) - min(time_list)))
links.sort(key =lambda x: x[2])
print ("# temporal links", len(links))

# prepare slicing
SLICE_INTERVAL = 2.5e5
START, END = min(time_list), max(time_list)

slices_links = defaultdict(lambda : nx.MultiGraph())
# perform slicing
print("Slicing starts...")
slice_id = 0

for (user_id, item_id, ts, user_label, feat) in links:
    prev_slice_id = slice_id
    curr_time = ts
    time_diff = curr_time - START
        
    slice_id = time_diff // SLICE_INTERVAL
    
    if slice_id == 1+prev_slice_id and slice_id > 0:
        # should be on next graph
        slices_links[slice_id] = nx.MultiGraph()
        slices_links[slice_id].add_nodes_from(slices_links[slice_id-1].nodes(data=True))
        assert(len(slices_links[slice_id].edges())==0)

    if slice_id == 1+prev_slice_id and slice_id == 0:
        # the very first graph
        slices_links[slice_id] = nx.MultiGraph()

    if user_id not in slices_links[slice_id]:
        slices_links[slice_id].add_node(user_id)
        # add user_id node label
        slices_links[slice_id].nodes[user_id]['label'] = user_label
    else:
        # update label
        slices_links[slice_id].nodes[user_id]['label'] = user_label

    if item_id not in slices_links[slice_id]:
        slices_links[slice_id].add_node(item_id)
        # add item_id node label as -1
        slices_links[slice_id].nodes[item_id]['label'] = -1

    slices_links[slice_id].add_edge(user_id, item_id, time=ts, feat=feat)

# slices_links: list of Multigraphs
# prepare slices_features <node feats>: not used in DGLC learning 
slices_features = defaultdict(lambda : {})

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
        for x in slices_graph[slice_id].nodes(data=True):
            G.add_node(node_idx[x[0]], label=x[1]['label'])
        for x in slices_graph[slice_id].edges(data=True):
            G.add_edge(node_idx[x[0]], node_idx[x[1]], feat=x[2]['feat'])
        assert (len(G.nodes()) == len(slices_graph[slice_id].nodes()))
        assert (len(G.edges()) == len(slices_graph[slice_id].edges()))
        slices_graph_remap.append(G)
    
    for slice_id in slices_features:
        slice_id = int(slice_id)
        features_remap = []
        for x in slices_graph_remap[slice_id].nodes():
            features_remap.append(slices_features[slice_id][idx_node[x]])
        features_remap = csr_matrix(np.squeeze(np.array(features_remap)))
        slices_features_remap.append(features_remap)
    return (slices_graph_remap, slices_features_remap)

slices_links_remap, slices_features_remap = remap(slices_links, slices_features)

np.savez('graphs.npz', graph=slices_links_remap)
np.savez('features.npz', feats=slices_features_remap)
