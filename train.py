import torch
import torch.optim as optim
import torch_geometric.nn as pyg_nn
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData # For type hinting
import os
from tqdm import tqdm

# Project imports
from data_loader import (
    load_steam_data,
    load_steam_description_data,
    load_steamspy_tag_data,
    preprocess_and_merge_data
)
from graph_builder import build_heterogeneous_graph
from gnn_model import HeteroGNN

# Configuration Variables
DATA_DIR = './data/' # Assuming data_loader.py uses this or its own DATA_PATH
MODEL_SAVE_PATH = './models/'
NUM_SYNTHETIC_USERS = 1000 # Should match what's used in graph_builder if simulating
SIMULATE_USERS_IN_TRAINING = True # Control if user simulation is part of graph building here

HIDDEN_CHANNELS = 64 # Reduced for faster example training, can be 128 or 256
NUM_GNN_LAYERS = 2  # For GNNEncoder or number of neighbor sampling hops
USE_GNN_ENCODER = False # Simpler model for first pass, can be True
LEARNING_RATE = 0.001 # Common starting LR
EPOCHS = 10 # Reduced for faster example, typically 50-200
BATCH_SIZE = 128 # Adjust based on GPU memory and dataset size

# Ensure model save path exists
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def get_node_feature_dims(data: HeteroData) -> dict:
    """
    Inspects the HeteroData object and returns a dictionary mapping
    node type to its feature dimension.
    """
    feature_dims = {}
    for node_type in data.node_types:
        if data[node_type].x is not None: # Check if features exist
            feature_dims[node_type] = data[node_type].x.shape[1]
        else: # Should ideally not happen for nodes involved in GNN
            print(f"Warning: Node type '{node_type}' has no features (data['{node_type}'].x is None). Setting dim to 0.")
            feature_dims[node_type] = 0
    return feature_dims

if __name__ == '__main__':
    print("--- Starting Training Script ---")

    # 1. Load and Prepare Data
    print("\n--- 1. Loading and Preparing Data ---")
    steam_df = load_steam_data(data_path=DATA_DIR)
    description_df = load_steam_description_data(data_path=DATA_DIR)
    tags_df = load_steamspy_tag_data(data_path=DATA_DIR)

    core_game_data_df = preprocess_and_merge_data(steam_df, description_df, tags_df)

    # Build graph (ensure simulate_users is True if training on user-game links)
    print(f"Building graph with simulate_users={SIMULATE_USERS_IN_TRAINING}, num_synthetic_users={NUM_SYNTHETIC_USERS}")
    graph_data, game_map, tag_map, dev_map, user_map = build_heterogeneous_graph(
        core_game_data_df,
        simulate_users=SIMULATE_USERS_IN_TRAINING,
        num_synthetic_users=NUM_SYNTHETIC_USERS
    )

    metadata = graph_data.metadata()
    node_feature_dims = get_node_feature_dims(graph_data)

    print(f"Graph metadata: {metadata}")
    print(f"Node feature dimensions: {node_feature_dims}")

    if ('user', 'likes', 'game') not in graph_data.edge_types:
        raise ValueError("Edge type ('user', 'likes', 'game') not found in graph. Cannot train for link prediction on this edge.")
    if graph_data['user', 'likes', 'game'].num_edges == 0:
         raise ValueError("No edges found for ('user', 'likes', 'game'). Check data simulation in graph_builder.")


    # Link Prediction Setup for ('user', 'likes', 'game')
    print("\nSetting up LinkNeighborLoader...")
    # The LinkNeighborLoader will sample neighbors and also provide negative samples.
    # Edge labels (1 for positive, 0 for negative) will be part of its output.
    train_loader = pyg_nn.LinkNeighborLoader(
        data=graph_data,
        num_neighbors=[HIDDEN_CHANNELS // 32] * NUM_GNN_LAYERS, # Example: sample ~2 neighbors per layer if HIDDEN_CHANNELS=64
        # num_neighbors=[-1] * NUM_GNN_LAYERS, # Sample all neighbors (can be slow/memory intensive)
        batch_size=BATCH_SIZE,
        edge_label_index=(('user', 'likes', 'game'), graph_data['user', 'likes', 'game'].edge_index),
        # edge_label is not strictly needed here if only positive edges are passed,
        # as LinkNeighborLoader by default assumes these are positive and samples negatives.
        # However, if you have pre-defined negative edges, you'd include them and their labels.
        shuffle=True,
        # neg_sampling_ratio can be set if specific ratio is desired, default is 1.0 (as many neg as pos)
    )
    print(f"LinkNeighborLoader initialized. Sample batch: {next(iter(train_loader))}")


    # 2. Initialize Model, Optimizer
    print("\n--- 2. Initializing Model and Optimizer ---")
    # Ensure all node types present in training edges have feature dimensions
    for node_type in ['user', 'game']: # Specific to ('user', 'likes', 'game')
        if node_type not in node_feature_dims or node_feature_dims[node_type] == 0:
            raise ValueError(f"Node type '{node_type}' has 0 feature dimension or is missing, but is required for 'user-likes-game' link prediction.")

    model = HeteroGNN(
        node_feature_dims=node_feature_dims,
        hidden_channels=HIDDEN_CHANNELS,
        num_gnn_layers=NUM_GNN_LAYERS, # This is for GNNEncoder, not directly used by LinkNeighborLoader's num_neighbors
        metadata=metadata,
        use_encoder=USE_GNN_ENCODER
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model initialized and moved to {device}:")
    print(model)

    # 3. Training Loop
    print("\n--- 3. Starting Training Loop ---")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_examples = 0 # To correctly average loss over actual number of examples processed

        # Wrap train_loader with tqdm for a progress bar
        for sampled_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            sampled_data = sampled_data.to(device)
            optimizer.zero_grad()

            # Get embeddings from the model using the sampled subgraph
            out_embeddings_dict = model(sampled_data) # Pass the HeteroData batch directly

            # Get user and game embeddings for the source and destination nodes of the links being predicted
            # The edge_label_index in sampled_data refers to nodes within that subgraph.
            # These nodes are already part of out_embeddings_dict.
            # Their original node IDs are in sampled_data['user'].n_id and sampled_data['game'].n_id
            # but pyg_nn.LinkNeighborLoader remaps them to be 0-indexed within the batch.

            user_src_nodes, game_dst_nodes = sampled_data['user', 'likes', 'game'].edge_label_index

            user_embeds = out_embeddings_dict['user'][user_src_nodes]
            game_embeds = out_embeddings_dict['game'][game_dst_nodes]

            # Predict scores (dot product)
            pred = (user_embeds * game_embeds).sum(dim=-1)

            target = sampled_data['user', 'likes', 'game'].edge_label

            loss = F.binary_cross_entropy_with_logits(pred, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * pred.size(0) # pred.size(0) is num_edges_in_batch
            total_examples += pred.size(0)

        if total_examples == 0:
            avg_loss = 0 # Avoid division by zero if loader is empty
        else:
            avg_loss = total_loss / total_examples

        print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")

    # 4. Save Model
    print("\n--- 4. Saving Model ---")
    model_path = os.path.join(MODEL_SAVE_PATH, 'hetero_gnn_link_pred_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model trained and saved to {model_path}")

    print("\n--- Training Script Finished ---")
