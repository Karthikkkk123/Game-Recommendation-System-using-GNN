import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import HeteroData # For type hinting and dummy data creation

# Optional helper module for a stack of GNN layers (e.g., SAGEConv)
class GNNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        # First layer can take specific in_channels or -1 for lazy init
        # Subsequent layers will take hidden_channels as input
        current_in_channels = in_channels
        for i in range(num_layers):
            # Using (-1, -1) for SAGEConv to handle bipartite graphs or lazy feature size inference
            conv = pyg_nn.SAGEConv((-1, -1), hidden_channels)
            self.convs.append(conv)
            # Input to next conv is output of current one (hidden_channels)
            # This isn't strictly needed if SAGEConv uses (-1,-1) but good for clarity if mixing convs
            # current_in_channels = hidden_channels

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x) # Apply ReLU after each convolution
            # Consider adding Dropout here if needed: x = F.dropout(x, p=0.5, training=self.training)
        return x

class HeteroGNN(torch.nn.Module):
    def __init__(self, node_feature_dims, hidden_channels, num_gnn_layers=2, metadata=None, use_encoder=False):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.use_encoder = use_encoder # Flag to decide if using GNNEncoder or single SAGEConv

        # Initial linear projection layers for each node type
        self.lins = torch.nn.ModuleDict()
        for node_type, in_dim in node_feature_dims.items():
            self.lins[node_type] = pyg_nn.Linear(in_dim, hidden_channels)

        # GNN part using to_hetero
        if self.use_encoder:
            # The GNNEncoder will expect hidden_channels as input after the initial projection
            # It will also output hidden_channels
            encoder_gnn = GNNEncoder(hidden_channels, hidden_channels, num_layers=num_gnn_layers)
            self.gnn = pyg_nn.to_hetero(encoder_gnn, metadata, aggr='sum')
        else:
            # Simpler: a single SAGEConv layer per edge type, adapted by to_hetero
            # Input to SAGEConv here will be hidden_channels (post-projection)
            # Output will also be hidden_channels
            base_gnn = pyg_nn.SAGEConv((-1, -1), hidden_channels)
            self.gnn = pyg_nn.to_hetero(base_gnn, metadata, aggr='sum')

    def forward(self, data: HeteroData):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        # 1. Initial Projection for each node type
        projected_x_dict = {}
        for node_type, x_val in x_dict.items():
            if node_type in self.lins: # Ensure projection layer exists for the node type
                 projected_x_dict[node_type] = self.lins[node_type](x_val).relu()
            else: # Should not happen if node_feature_dims is comprehensive
                 projected_x_dict[node_type] = x_val


        # 2. GNN Pass (handles heterogeneous message passing)
        # The GNNEncoder (if used) or SAGEConv expects (x_src, x_dst) for bipartite,
        # or just x for homogeneous. to_hetero handles passing the correct projected features.
        embeddings_dict = self.gnn(projected_x_dict, edge_index_dict)

        return embeddings_dict

if __name__ == '__main__':
    print("--- Testing HeteroGNN Model ---")

    # 1. Create a dummy HeteroData object
    data = HeteroData()
    num_games = 10
    game_feat_dim = 32
    num_tags = 5
    tag_feat_dim = 16
    num_users = 7
    user_feat_dim = 8 # Example: if users had features

    data['game'].x = torch.randn(num_games, game_feat_dim)
    data['tag'].x = torch.randn(num_tags, tag_feat_dim)
    data['user'].x = torch.randn(num_users, user_feat_dim) # Add user features

    # Game-Tag edges
    data['game', 'has_tag', 'tag'].edge_index = torch.randint(0, max(num_games, num_tags), (2, 20), dtype=torch.long)
    # Ensure indices are within bounds:
    data['game', 'has_tag', 'tag'].edge_index[0] %= num_games
    data['game', 'has_tag', 'tag'].edge_index[1] %= num_tags

    # User-Game "likes" edges
    data['user', 'likes', 'game'].edge_index = torch.randint(0, max(num_users, num_games), (2, 15), dtype=torch.long)
    data['user', 'likes', 'game'].edge_index[0] %= num_users
    data['user', 'likes', 'game'].edge_index[1] %= num_games

    # Game-Game "similarity" edges
    data['game', 'similar_to', 'game'].edge_index = torch.randint(0, num_games, (2, 25), dtype=torch.long)
    # Remove self-loops for similarity for this test
    sel = data['game', 'similar_to', 'game'].edge_index[0] != data['game', 'similar_to', 'game'].edge_index[1]
    data['game', 'similar_to', 'game'].edge_index = data['game', 'similar_to', 'game'].edge_index[:, sel]


    print("\nDummy HeteroData object created:")
    print(data)

    # 2. Define node_feature_dims for the model
    node_feature_dims = {
        'game': game_feat_dim,
        'tag': tag_feat_dim,
        'user': user_feat_dim
    }
    hidden_channels = 64

    # --- Test with single SAGEConv wrapped by to_hetero ---
    print(f"\n--- Testing with use_encoder=False (single SAGEConv adapted by to_hetero) ---")
    try:
        model_simple = HeteroGNN(
            node_feature_dims=node_feature_dims,
            hidden_channels=hidden_channels,
            metadata=data.metadata(),
            use_encoder=False
        )
        print("\nHeteroGNN (use_encoder=False) instantiated:")
        print(model_simple)

        output_embeddings_simple = model_simple(data)
        print("\nOutput Embeddings (use_encoder=False):")
        for node_type, embedding in output_embeddings_simple.items():
            print(f"  Node type '{node_type}': Output shape {embedding.shape}")
            assert embedding.shape == (data[node_type].num_nodes, hidden_channels)
        print("Test with use_encoder=False PASSED.")
    except Exception as e:
        print(f"ERROR during test with use_encoder=False: {e}")
        import traceback
        traceback.print_exc()


    # --- Test with GNNEncoder wrapped by to_hetero ---
    print(f"\n--- Testing with use_encoder=True (GNNEncoder adapted by to_hetero) ---")
    try:
        model_encoder = HeteroGNN(
            node_feature_dims=node_feature_dims, # Input to initial Linear layers
            hidden_channels=hidden_channels,    # Output of Linear, input/hidden/output of GNNEncoder
            num_gnn_layers=2,
            metadata=data.metadata(),
            use_encoder=True
        )
        print("\nHeteroGNN (use_encoder=True) instantiated:")
        print(model_encoder)

        output_embeddings_encoder = model_encoder(data)
        print("\nOutput Embeddings (use_encoder=True):")
        for node_type, embedding in output_embeddings_encoder.items():
            print(f"  Node type '{node_type}': Output shape {embedding.shape}")
            assert embedding.shape == (data[node_type].num_nodes, hidden_channels)
        print("Test with use_encoder=True PASSED.")
    except Exception as e:
        print(f"ERROR during test with use_encoder=True: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- HeteroGNN Model Test Script Finished ---")
