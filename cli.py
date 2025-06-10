import argparse
import torch
import os

# Project imports
from inference import (
    load_inference_data_and_mappings,
    get_recommendations
    # get_node_feature_dims is called within load_inference_data_and_mappings
)
from gnn_model import HeteroGNN # For model instantiation

# Default Configuration (align these with your training defaults and saved model)
DEFAULT_MODEL_FILENAME = 'hetero_gnn_link_pred_model.pt'
DEFAULT_MODEL_DIR = './models/'
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, DEFAULT_MODEL_FILENAME)

DEFAULT_TOP_N = 10

# Model architecture defaults - THESE MUST MATCH THE SAVED MODEL
DEFAULT_HIDDEN_CHANNELS = 64  # As used in train.py and inference.py examples
DEFAULT_NUM_GNN_LAYERS = 2    # As used in train.py and inference.py examples
DEFAULT_USE_GNN_ENCODER = False # As used in train.py and inference.py examples

# Data loading defaults
DEFAULT_DATA_DIR = './data/'
DEFAULT_NUM_SYNTHETIC_USERS = 1000 # Must match graph structure if model metadata depends on it
DEFAULT_SIMULATE_USERS_FOR_GRAPH = True # Must match graph structure if model metadata depends on it


def main():
    parser = argparse.ArgumentParser(description="Get game recommendations based on a list of liked games.")
    parser.add_argument('game_names', metavar='GAME_NAME', type=str, nargs='+',
                        help='One or more game names you like.')
    parser.add_argument('--top_n', type=int, default=DEFAULT_TOP_N,
                        help=f'Number of recommendations to return (default: {DEFAULT_TOP_N}).')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help=f'Path to the trained GNN model file (default: {DEFAULT_MODEL_PATH}).')

    # Arguments for model architecture - user must ensure these match the loaded model
    parser.add_argument('--hidden_channels', type=int, default=DEFAULT_HIDDEN_CHANNELS,
                        help=f'Hidden channels of the GNN model (default: {DEFAULT_HIDDEN_CHANNELS}). Must match saved model.')
    parser.add_argument('--num_gnn_layers', type=int, default=DEFAULT_NUM_GNN_LAYERS,
                        help=f'Number of GNN layers in the GNNEncoder (if used) (default: {DEFAULT_NUM_GNN_LAYERS}). Must match saved model.')
    parser.add_argument('--use_gnn_encoder', action=argparse.BooleanOptionalAction, default=DEFAULT_USE_GNN_ENCODER, # Allows --use_gnn_encoder or --no-use_gnn_encoder
                        help=f'Whether the GNN model uses the GNNEncoder structure (default: {DEFAULT_USE_GNN_ENCODER}). Must match saved model.')

    # Arguments for data loading configuration
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR, help=f'Directory containing the data files (default: {DEFAULT_DATA_DIR})')
    parser.add_argument('--num_synthetic_users', type=int, default=DEFAULT_NUM_SYNTHETIC_USERS, help=f'Number of synthetic users for graph construction (default: {DEFAULT_NUM_SYNTHETIC_USERS})')
    parser.add_argument('--simulate_users', action=argparse.BooleanOptionalAction, default=DEFAULT_SIMULATE_USERS_FOR_GRAPH, help=f'Simulate users during graph construction (default: {DEFAULT_SIMULATE_USERS_FOR_GRAPH})')


    args = parser.parse_args()

    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Please ensure the path is correct and the model is trained.")
        return

    # Load Data and Mappings
    # load_inference_data_and_mappings from inference.py handles its own data path and user simulation config
    # but we can allow overrides via CLI if desired, for now using defaults set in inference.py or passed here.
    print("Loading data, building graph, and creating mappings...")
    # The inference.py load_inference_data_and_mappings uses its own defaults for these,
    # but let's pass them from CLI to show how it could be configured.
    # The inference.py script uses constants like DATA_DIR, SIMULATE_USERS_FOR_GRAPH, NUM_SYNTHETIC_USERS
    # For this CLI, we use the args.data_dir, args.simulate_users, args.num_synthetic_users
    # We need to modify load_inference_data_and_mappings to accept these, or set globals in inference.py
    # For now, let's assume inference.py's constants are used, and these CLI args are for future flexibility
    # or if we refactor load_inference_data_and_mappings to accept them.
    # For this exercise, we'll assume the defaults in this CLI script control it for consistency if inference.py were refactored.
    # The inference.py script's `load_inference_data_and_mappings` does not currently accept these as args.
    # For now, it will use the constants defined within inference.py.
    # This is a limitation of the current `inference.py` design if we want CLI to fully control those.
    # Let's proceed assuming `inference.py` uses its internal defaults for data loading params.

    # To make this work as intended, load_inference_data_and_mappings in inference.py would need to be:
    # def load_inference_data_and_mappings(data_dir_param=DATA_DIR, simulate_users_param=SIMULATE_USERS_FOR_GRAPH, ...):
    # For now, it will use its own internal constants. The CLI args are illustrative.
    graph_data, game_id_map, id_to_game_name_map, name_to_steam_appid, node_feature_dims = \
        load_inference_data_and_mappings() # Uses constants from inference.py for data path etc.
                                           # Ideally, these would be passed: (data_dir=args.data_dir, ...)

    graph_data = graph_data.to(device)
    print("Data loaded and graph moved to device.")

    # Load Model
    print(f"Loading model from {args.model_path}...")
    # node_feature_dims is derived from graph_data
    # metadata is derived from graph_data
    model = HeteroGNN(
        node_feature_dims=node_feature_dims,
        hidden_channels=args.hidden_channels,
        num_gnn_layers=args.num_gnn_layers,
        metadata=graph_data.metadata(),
        use_encoder=args.use_gnn_encoder
    )

    # Load state dict to CPU first, then move model, to avoid GPU OOM if state_dict is large
    state_dict = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("Model loaded and set to evaluation mode.")

    # Get and Display Recommendations
    print(f"\nGetting recommendations for: {', '.join(args.game_names)} (Top {args.top_n})")

    recommendations = get_recommendations(
        input_game_names=args.game_names,
        model=model,
        graph_data=graph_data, # Already on device
        game_id_map=game_id_map,
        id_to_game_name_map=id_to_game_name_map,
        name_to_steam_appid=name_to_steam_appid,
        device=device, # Passed to get_recommendations for any internal tensor ops
        top_n=args.top_n
    )

    if recommendations:
        print(f"\n--- Top {args.top_n} Recommendations ---")
        for i, (name, score) in enumerate(recommendations):
            print(f"{i+1}. {name} (Similarity: {score:.4f})")
    else:
        print("\nCould not generate recommendations.")
        print("This might be because the input game names were not found in the dataset,")
        print("or no similar games were found based on the model's criteria.")

if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print(f"Error: A required file was not found.")
        print(f"Details: {e}")
        print("Please ensure all data files and model checkpoints are in their expected locations.")
    except Exception as e:
        print(f"An unexpected error occurred:")
        import traceback
        traceback.print_exc()

    print("\nCLI finished.")
