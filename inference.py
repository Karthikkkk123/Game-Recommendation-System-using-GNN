import torch
import torch.nn.functional as F
import pandas as pd
import os
import numpy as np

# Project imports
from gnn_model import HeteroGNN
from data_loader import (
    load_steam_data,
    load_steam_description_data,
    load_steamspy_tag_data,
    preprocess_and_merge_data
)
from graph_builder import build_heterogeneous_graph # To reconstruct graph for inference
# from train import get_node_feature_dims # Option to import

# Configuration Variables
MODEL_FILENAME = 'hetero_gnn_link_pred_model.pt' # Make sure this matches the saved model name
MODEL_PATH = os.path.join('./models/', MODEL_FILENAME)
DATA_DIR = './data/'

# These should match the parameters used during training for the saved model
# If these are different, the loaded model architecture won't match the state_dict
NUM_SYNTHETIC_USERS = 1000 # Must match if graph structure depends on it for model metadata
SIMULATE_USERS_FOR_GRAPH = True # Must match if graph structure depends on it

HIDDEN_CHANNELS = 64 # From train.py example
NUM_GNN_LAYERS = 2   # From train.py example
USE_GNN_ENCODER = False # From train.py example

TOP_N_RECOMMENDATIONS = 10

# Replicating get_node_feature_dims from train.py for self-containment or if train.py is not available
def get_node_feature_dims(data: 'HeteroData') -> dict: # Use string for HeteroData due to potential circular import
    """
    Inspects the HeteroData object and returns a dictionary mapping
    node type to its feature dimension.
    """
    feature_dims = {}
    for node_type in data.node_types:
        if data[node_type].x is not None:
            feature_dims[node_type] = data[node_type].x.shape[1]
        else:
            print(f"Warning: Node type '{node_type}' has no features. Setting dim to 0.")
            feature_dims[node_type] = 0
    return feature_dims

def load_inference_data_and_mappings():
    print("--- Loading Inference Data and Building Graph ---")
    # Load steam.csv for appid-name mapping
    steam_main_df = load_steam_data(data_path=DATA_DIR) # steam_df in data_loader renames appid
    # The load_steam_data renames 'appid' to 'steam_appid', so use 'steam_appid'
    name_to_steam_appid = pd.Series(steam_main_df.steam_appid.values, index=steam_main_df.name).to_dict()

    # Load other data required for graph construction (as done in train.py)
    description_df = load_steam_description_data(data_path=DATA_DIR)
    tags_df = load_steamspy_tag_data(data_path=DATA_DIR)

    core_game_data_df = preprocess_and_merge_data(steam_main_df, description_df, tags_df)

    graph_data, game_id_map, tag_id_map, dev_id_map, user_id_map = build_heterogeneous_graph(
        core_game_data_df,
        simulate_users=SIMULATE_USERS_FOR_GRAPH, # Consistent with how model might have seen metadata
        num_synthetic_users=NUM_SYNTHETIC_USERS
    )

    node_feature_dims = get_node_feature_dims(graph_data)

    # Create reverse mapping from game integer node ID to steam_appid, then to name
    id_to_steam_appid = {v: k for k, v in game_id_map.items()}

    id_to_game_name_map = {}
    # steam_main_df here should be the one from load_steam_data, which has 'steam_appid' and 'name'
    temp_steam_df_for_names = steam_main_df[['steam_appid', 'name']].copy()
    temp_steam_df_for_names.set_index('steam_appid', inplace=True)

    for node_id, steam_appid_val in id_to_steam_appid.items():
        try:
            id_to_game_name_map[node_id] = temp_steam_df_for_names.loc[steam_appid_val, 'name']
        except KeyError:
            # This might happen if a steam_appid in game_id_map is not in steam_main_df
            # (e.g. due to filtering in core_game_data_df that game_id_map is based on)
            # Or if steam_main_df was not the result of load_steam_data()
            print(f"Warning: steam_appid {steam_appid_val} (Node ID {node_id}) not found in name mapping source. Skipping.")
            id_to_game_name_map[node_id] = f"Unknown Game (AppID: {steam_appid_val})"


    print(f"Loaded graph with {graph_data.num_nodes} total nodes.")
    print(f"Game ID map size: {len(game_id_map)}")
    print(f"ID to Game Name map size: {len(id_to_game_name_map)}")

    return graph_data, game_id_map, id_to_game_name_map, name_to_steam_appid, node_feature_dims


all_node_embeddings_cache = None # Global cache for all node embeddings

def get_recommendations(
    input_game_names: list[str],
    model: HeteroGNN,
    graph_data: 'HeteroData',
    game_id_map: dict,
    id_to_game_name_map: dict,
    name_to_steam_appid: dict,
    device: torch.device,
    top_n: int = 10
):
    global all_node_embeddings_cache
    model.eval() # Set model to evaluation mode

    input_game_node_ids = []
    valid_input_game_names = []
    for name in input_game_names:
        steam_appid = name_to_steam_appid.get(name)
        if steam_appid:
            node_id = game_id_map.get(steam_appid)
            if node_id is not None:
                input_game_node_ids.append(node_id)
                valid_input_game_names.append(name)
            else:
                print(f"Warning: Game '{name}' (AppID {steam_appid}) not found in game_id_map (graph nodes). Skipping.")
        else:
            print(f"Warning: Game name '{name}' not found in name_to_steam_appid map. Skipping.")

    if not input_game_node_ids:
        print("No valid input games found to base recommendations on.")
        return []

    print(f"Generating recommendations based on: {valid_input_game_names}")

    # Generate or retrieve all node embeddings
    if all_node_embeddings_cache is None:
        print("Calculating all node embeddings (first time or cache cleared)...")
        with torch.no_grad():
            # Move graph_data to device if not already done (depends on where it was constructed)
            # For inference, if graph_data is large, this could be an issue.
            # Typically, graph_data for model forward pass should be on the same device as model.
            graph_data = graph_data.to(device)
            all_node_embeddings_cache = model(graph_data)
        print("All node embeddings calculated and cached.")

    game_embeddings_tensor = all_node_embeddings_cache['game'] # Assuming 'game' is the key for game embeddings

    # Aggregate input game embeddings
    input_game_embeddings = game_embeddings_tensor[torch.tensor(input_game_node_ids, device=device)]

    if input_game_embeddings.ndim == 1: # Should not happen if game_embeddings_tensor is (num_games, feat_dim)
        user_profile_embedding = input_game_embeddings
    elif input_game_embeddings.shape[0] > 1:
        user_profile_embedding = input_game_embeddings.mean(dim=0)
    else: # Single game input
        user_profile_embedding = input_game_embeddings.squeeze(0)

    # Compute cosine similarities
    # Ensure user_profile_embedding is 2D for batch comparison with game_embeddings_tensor if it's not already
    if user_profile_embedding.ndim == 1:
      user_profile_embedding = user_profile_embedding.unsqueeze(0)

    similarities = F.cosine_similarity(user_profile_embedding, game_embeddings_tensor)

    # Get Top-N recommendations
    # Exclude input games from recommendations
    # Create a mask for input game node IDs to ignore them in similarity results
    # We have game_embeddings_tensor which is (num_all_games, feat_dim)
    # Similarities is (num_all_games)
    # We need to set similarities of input_game_node_ids to a very low value before sorting

    # Make a copy to avoid modifying the original similarities tensor if it's used elsewhere
    masked_similarities = similarities.clone()
    for node_id in input_game_node_ids:
        # game_id_map maps steam_appid to node_id.
        # game_embeddings_tensor is indexed by node_id if node_ids are contiguous from 0.
        # So, if node_id is directly the index in game_embeddings_tensor:
        masked_similarities[node_id] = -float('inf') # Set to very low value

    # Get top N scores and their indices
    top_scores, top_indices = torch.topk(masked_similarities, top_n)

    recommendations = []
    for i in range(top_indices.size(0)):
        game_node_id = top_indices[i].item()
        score = top_scores[i].item()
        game_name = id_to_game_name_map.get(game_node_id, f"Unknown Game (Node ID: {game_node_id})")
        recommendations.append((game_name, score))

    return recommendations

if __name__ == '__main__':
    print("--- Inference Script Started ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        graph_data, game_id_map, id_to_game_name_map, name_to_steam_appid, node_feature_dims = load_inference_data_and_mappings()

        # Instantiate model
        model = HeteroGNN(
            node_feature_dims=node_feature_dims,
            hidden_channels=HIDDEN_CHANNELS,
            num_gnn_layers=NUM_GNN_LAYERS,
            metadata=graph_data.metadata(), # Critical for to_hetero
            use_encoder=USE_GNN_ENCODER
        )

        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model file not found at {MODEL_PATH}")
            print("Please ensure the model is trained and saved correctly.")
            exit()

        # Load trained model weights
        # model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        # It's generally safer to load state dict to CPU first, then move model to device
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model = model.to(device) # Move model to device
        model.eval() # Set to evaluation mode
        print(f"Model loaded from {MODEL_PATH} and set to evaluation mode.")

        # Example usage:
        # Use game names that you know are in your steam.csv
        # Example: liked_games = ["Counter-Strike", "Half-Life", "Team Fortress Classic"]
        # Use a short list for testing:
        liked_games = ["Counter-Strike", "Team Fortress Classic"]
        # Check if these games are in name_to_steam_appid:
        print(f"\nLooking up example liked games: {liked_games}")
        for g_name in liked_games:
            if g_name not in name_to_steam_appid:
                print(f"  Warning: Example game '{g_name}' is not in the name_to_steam_appid map. It will be skipped.")
                print(f"  Consider using game names directly from your 'steam.csv' for testing.")

        recommendations = get_recommendations(
            input_game_names=liked_games,
            model=model,
            graph_data=graph_data, # graph_data will be moved to device inside get_recommendations if needed
            game_id_map=game_id_map,
            id_to_game_name_map=id_to_game_name_map,
            name_to_steam_appid=name_to_steam_appid,
            device=device,
            top_n=TOP_N_RECOMMENDATIONS
        )

        print(f"\n--- Top {TOP_N_RECOMMENDATIONS} Recommendations for {liked_games} ---")
        if recommendations:
            for i, (name, score) in enumerate(recommendations):
                print(f"{i+1}. {name} (Similarity: {score:.4f})")
        else:
            print("No recommendations generated.")

    except FileNotFoundError as e:
        print(f"ERROR: Could not find necessary data files. {e}")
        print("Please ensure 'steam.csv' and other required CSVs are in the DATA_DIR.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Inference Script Finished ---")
