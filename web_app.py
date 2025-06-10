from flask import Flask, render_template, request, url_for
import torch
import os

# Project imports
from inference import (
    load_inference_data_and_mappings,
    get_recommendations
)
from gnn_model import HeteroGNN # For model instantiation

# Configuration (ensure these match your trained model and setup)
MODEL_FILENAME = 'hetero_gnn_link_pred_model.pt' # Ensure this is the correct name
MODEL_DIR = './models/'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

DATA_DIR = './data/' # Used by load_inference_data_and_mappings

# Model architecture parameters - MUST MATCH THE SAVED MODEL
HIDDEN_CHANNELS = 64  # As used in train.py example
NUM_GNN_LAYERS = 2    # As used in train.py example
USE_GNN_ENCODER = False # As used in train.py example

# Graph construction parameters (if they affect metadata seen by the model)
NUM_SYNTHETIC_USERS = 1000 # Must match graph structure if model metadata depends on it
SIMULATE_USERS_FOR_GRAPH = True # Must match graph structure if model metadata depends on it

TOP_N_RECOMMENDATIONS = 10

# Global variables for model and data (initialized to None)
# These will be loaded once when the app starts.
model_instance = None
graph_data_instance = None
game_id_map_instance = None
id_to_game_name_map_instance = None
name_to_steam_appid_instance = None
node_feature_dims_instance = None
device_instance = None

def initialize_app_resources():
    """Loads the model, data, and mappings once when the app starts."""
    global model_instance, graph_data_instance, game_id_map_instance, \
           id_to_game_name_map_instance, name_to_steam_appid_instance, \
           node_feature_dims_instance, device_instance

    print("Initializing Flask app resources...")

    device_instance = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device_instance}")

    if not os.path.exists(MODEL_PATH):
        error_message = f"FATAL ERROR: Model file not found at {MODEL_PATH}. The web application cannot start."
        print(error_message)
        raise FileNotFoundError(error_message)

    # load_inference_data_and_mappings is assumed to use its own internal constants for
    # data_dir, simulate_users, num_synthetic_users if not passed as arguments.
    # For this app, we rely on the constants defined in inference.py or here if it were refactored.
    # The function signature in inference.py would need to accept these to be configurable from here.
    # For now, we use the constants defined at the top of this file for clarity on what this app *expects*.
    # load_inference_data_and_mappings in inference.py currently uses its *own* constants.
    # This is a point of potential mismatch if not managed carefully.
    graph_data, game_id_map, id_to_game_name_map, name_to_steam_appid, node_feature_dims = \
        load_inference_data_and_mappings() # This will use constants defined in inference.py

    graph_data_instance = graph_data.to(device_instance)
    game_id_map_instance = game_id_map
    id_to_game_name_map_instance = id_to_game_name_map
    name_to_steam_appid_instance = name_to_steam_appid
    node_feature_dims_instance = node_feature_dims

    print("Instantiating GNN model...")
    model = HeteroGNN(
        node_feature_dims=node_feature_dims_instance,
        hidden_channels=HIDDEN_CHANNELS,
        num_gnn_layers=NUM_GNN_LAYERS,
        metadata=graph_data_instance.metadata(), # Critical for to_hetero
        use_encoder=USE_GNN_ENCODER
    )

    print(f"Loading model state_dict from {MODEL_PATH}...")
    # Load state dict to CPU first for safety, then move model to device
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model_instance = model.to(device_instance)
    model_instance.eval() # Set to evaluation mode

    print("Model and data loaded successfully for Flask app.")

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    if model_instance is None or graph_data_instance is None:
        # This case should ideally not be reached if initialize_app_resources works
        return render_template('recommendations.html', recommendations=None, input_games_str="N/A",
                               error_message="Application resources not initialized. Please check server logs.")

    input_text = request.form.get('game_names', '')
    input_game_names = [name.strip() for name in input_text.split(',') if name.strip()]

    if not input_game_names:
        return render_template('recommendations.html', recommendations=None,
                               input_games_str="No games provided",
                               error_message="Please enter at least one game name.")

    print(f"Received request for games: {input_game_names}")
    recommendations_list = get_recommendations(
        input_game_names=input_game_names,
        model=model_instance,
        graph_data=graph_data_instance, # Already on device
        game_id_map=game_id_map_instance,
        id_to_game_name_map=id_to_game_name_map_instance,
        name_to_steam_appid=name_to_steam_appid_instance,
        device=device_instance, # For any tensor ops within get_recommendations
        top_n=TOP_N_RECOMMENDATIONS
    )

    return render_template('recommendations.html',
                           recommendations=recommendations_list,
                           input_games_str=', '.join(input_game_names))

if __name__ == '__main__':
    try:
        initialize_app_resources()
        # Note: For production, use a proper WSGI server like Gunicorn or Waitress
        app.run(debug=True, host='0.0.0.0', port=5000)
    except FileNotFoundError as e:
        print(f"Failed to initialize application resources: {e}")
        print("The application cannot start without the model or essential data files.")
    except Exception as e:
        print(f"An unexpected error occurred during application startup: {e}")
        import traceback
        traceback.print_exc()
