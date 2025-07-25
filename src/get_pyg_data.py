import pandas as pd
from torch_geometric.data import Data
import torch


def load_bipartitedata(df_data=None):
    if df_data is None:
        df_data = pd.read_csv('../data/load/full_review.csv')
    user_node_ids = 0
    user_node_id_dict = dict()
    artist_node_ids = 0
    artist_node_id_dict = dict()
    user_reverse_dict = dict()
    artist_reverse_dict = dict()
    edge_data_u2a = []
    edge_data_a2u = []
    for index, row in df_data.iterrows():
        if str(row['user_id']) not in user_node_id_dict.keys():
            user_node_id_dict[str(row['user_id'])] = user_node_ids
            user_reverse_dict[user_node_ids] = row['user_id']
            user_node_ids = user_node_ids + 1
        if str(row['product_id']) not in artist_node_id_dict.keys():
            artist_node_id_dict[str(row['product_id'])] = artist_node_ids
            artist_reverse_dict[artist_node_ids] = row['product_id']
            artist_node_ids = artist_node_ids + 1
        edge_data_u2a.append([user_node_id_dict[str(row['user_id'])], artist_node_id_dict[str(row['product_id'])]])
        edge_data_a2u.append([artist_node_id_dict[str(row['product_id'])], user_node_id_dict[str(row['user_id'])]])


    return BipartiteData(edge_index_u2a=torch.LongTensor(edge_data_u2a).t().contiguous(),
                         edge_index_a2u=torch.LongTensor(edge_data_a2u).t().contiguous(),
                         num_users=user_node_ids, num_artists=artist_node_ids).to('cuda'), \
           user_reverse_dict, artist_reverse_dict, user_node_id_dict, artist_node_id_dict


class BipartiteData(Data):
    def __init__(self, edge_index_u2a=None, edge_index_a2u=None, num_artists=None, num_users=None):
        super().__init__()
        self.edge_index_u2a = edge_index_u2a
        self.edge_index_a2u = edge_index_a2u
        self.num_users = num_users
        self.num_artists = num_artists

    def __inc__(self, key, value, *args, **kwargs):
        # Returns the incremental count to cumulatively increase the value
        # of the next attribute of :obj:`key` when creating batches.
        if key == 'edge_index_u2a':
            return torch.tensor([[self.num_users], [self.num_artists]])
        elif key == 'edge_index_a2u':
            return torch.tensor([[self.num_artists], [self.num_users]])
        else:
            return super(BipartiteData, self).__inc__(key, value)