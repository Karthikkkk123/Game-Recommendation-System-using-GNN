import os
import json
import torch
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model and embeddings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the game data
df = pd.read_csv('data/steam.csv')
game_titles = df['name'].tolist()

def load_model_and_embeddings():
    try:
        # Load the game data and create title to index mapping
        df = pd.read_csv('data/steam.csv')
        game_titles = df['name'].tolist()
        game_to_idx = {game: idx for idx, game in enumerate(game_titles)}
        
        # Load the game embeddings
        game_embeddings = torch.load('gcn_game_embeddings.pt', map_location=DEVICE)
        
        # Since we don't have the original model architecture, we'll use the embeddings directly
        # for similarity calculations
        
        print(f"Successfully loaded embeddings for {len(game_titles)} games")
        return game_embeddings, game_titles, game_to_idx
        
    except Exception as e:
        import traceback
        print(f"Error loading model or embeddings: {e}")
        print(traceback.format_exc())
        return None, None, None

# Initialize embeddings and game data
game_embeddings, game_titles, game_to_idx = load_model_and_embeddings()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/search', methods=['GET'])
def search_games():
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify([])
    
    # Simple search - you might want to make this more sophisticated
    matches = [game for game in game_titles if query in game.lower()][:10]
    return jsonify(matches)

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    try:
        if game_embeddings is None or game_titles is None or game_to_idx is None:
            return jsonify({"error": "Model not loaded properly"}), 500
            
        data = request.get_json()
        selected_games = data.get('games', [])
        num_recommendations = min(10, data.get('num_recommendations', 5))
        
        if not selected_games:
            return jsonify({"error": "No games selected"}), 400
            
        # Get indices of selected games
        selected_indices = [game_to_idx[game] for game in selected_games if game in game_to_idx]
        
        if not selected_indices:
            return jsonify({"error": "No valid games found"}), 400
        
        # Convert indices to tensor if they're not already
        if not isinstance(selected_indices, torch.Tensor):
            selected_indices = torch.tensor(selected_indices, device=DEVICE)
        
        # Get embeddings for selected games and compute mean
        with torch.no_grad():
            selected_embeddings = game_embeddings[selected_indices].mean(dim=0, keepdim=True)
            
            # Calculate cosine similarity with all games
            similarities = torch.nn.functional.cosine_similarity(
                game_embeddings, 
                selected_embeddings.expand(game_embeddings.size(0), -1),
                dim=1
            )
        
        # Get top recommendations (excluding selected games)
        top_values, top_indices = torch.topk(similarities, k=len(similarities), sorted=True)
        
        recommendations = []
        for idx in top_indices.tolist():
            game_title = game_titles[idx]
            if game_title not in selected_games and game_title not in recommendations:
                recommendations.append(game_title)
                if len(recommendations) >= num_recommendations:
                    break
        
        return jsonify({"recommendations": recommendations})
        
    except Exception as e:
        import traceback
        print(f"Error in get_recommendations: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
