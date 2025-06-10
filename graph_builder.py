import torch
from torch_geometric.data import Data, HeteroData
import pandas as pd
import random

# Assuming data_loader.py is in the same directory or accessible in PYTHONPATH
from data_loader import (
    load_steam_data,
    load_steam_description_data,
    load_steamspy_tag_data,
    preprocess_and_merge_data
)

def create_game_mappings(core_game_data_df):
    """Creates a mapping from steam_appid to a unique integer node ID."""
    unique_appids = core_game_data_df['steam_appid'].unique()
    game_id_map = {appid: i for i, appid in enumerate(unique_appids)}
    num_games = len(unique_appids)
    print(f"Created mapping for {num_games} unique games.")
    return game_id_map, num_games

def create_tag_mappings_and_features(core_game_data_df):
    """Creates a mapping from tag string to a unique integer node ID and identity features."""
    all_tags = set()
    # Ensure 'steamspy_tag_list' exists and is a list
    if 'steamspy_tag_list' not in core_game_data_df.columns:
        print("Warning: 'steamspy_tag_list' column not found. No tags will be processed.")
        return {}, 0, torch.empty(0,0)

    for tag_list in core_game_data_df['steamspy_tag_list']:
        if isinstance(tag_list, list):
            for tag in tag_list:
                all_tags.add(tag)

    sorted_tags = sorted(list(all_tags))
    tag_id_map = {tag: i for i, tag in enumerate(sorted_tags)}
    num_tags = len(sorted_tags)
    tag_features = torch.eye(num_tags) if num_tags > 0 else torch.empty(0,0)
    print(f"Created mapping for {num_tags} unique tags and identity features of shape {tag_features.shape}.")
    return tag_id_map, num_tags, tag_features

def create_developer_mappings_and_features(core_game_data_df):
    """Creates a mapping from developer string to a unique integer node ID and identity features."""
    if 'developer' not in core_game_data_df.columns:
        print("Warning: 'developer' column not found. Proceeding without developer nodes/edges/features.")
        return {}, 0, torch.empty(0,0)

    unique_developers = core_game_data_df['developer'].dropna().unique()
    sorted_developers = sorted(list(unique_developers))
    dev_id_map = {dev: i for i, dev in enumerate(sorted_developers)}
    num_devs = len(sorted_developers)
    dev_features = torch.eye(num_devs) if num_devs > 0 else torch.empty(0,0)
    print(f"Created mapping for {num_devs} unique developers and identity features of shape {dev_features.shape}.")
    return dev_id_map, num_devs, dev_features

def create_user_mappings_and_features(num_synthetic_users):
    """Creates mappings and identity features for synthetic users."""
    user_id_map = {i: i for i in range(num_synthetic_users)}
    user_features = torch.eye(num_synthetic_users)
    print(f"Created mapping for {num_synthetic_users} synthetic users and identity features of shape {user_features.shape}.")
    return user_id_map, num_synthetic_users, user_features

def create_game_features(core_game_data_df, game_id_map, tag_id_map, num_tags):
    """Creates multi-hot encoded features for games based on their steamspy_tag_list."""
    if 'steamspy_tag_list' not in core_game_data_df.columns or num_tags == 0:
        print("Warning: 'steamspy_tag_list' column missing or no tags found. Game features will be empty.")
        return torch.empty(len(game_id_map), 0)

    num_games = len(game_id_map)
    game_features_list = []

    # Create a temporary Series with game_node_id for faster lookup if needed, though direct iteration is fine
    # appid_to_nodeid = pd.Series(game_id_map)

    for steam_appid, game_node_id in game_id_map.items():
        # Ensure correct row selection for the current game_node_id
        # We need to map steam_appid to its original row in core_game_data_df
        row = core_game_data_df[core_game_data_df['steam_appid'] == steam_appid].iloc[0]

        feature_vector = torch.zeros(num_tags, dtype=torch.float)
        if isinstance(row['steamspy_tag_list'], list):
            for tag_str in row['steamspy_tag_list']:
                tag_node_id = tag_id_map.get(tag_str)
                if tag_node_id is not None:
                    feature_vector[tag_node_id] = 1.0
        game_features_list.append(feature_vector)

    if not game_features_list: # If no games or no features generated
        return torch.empty(num_games, num_tags if num_tags > 0 else 0)

    game_features = torch.stack(game_features_list)
    print(f"Created game features of shape {game_features.shape}.")
    return game_features

def jaccard_similarity(set1, set2):
    """Computes Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0  # Or 0.0, depending on desired behavior for two empty sets
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0

def build_heterogeneous_graph(core_game_data_df, similarity_threshold=0.3, simulate_users=False, num_synthetic_users=1000):
    """Builds a heterogeneous graph, optionally including synthetic users and interactions."""
    data = HeteroData()
    user_id_map = None # Initialize to None

    game_id_map, num_games = create_game_mappings(core_game_data_df)
    tag_id_map, num_tags, tag_features = create_tag_mappings_and_features(core_game_data_df)
    dev_id_map, num_devs, dev_features = create_developer_mappings_and_features(core_game_data_df)

    game_features = create_game_features(core_game_data_df, game_id_map, tag_id_map, num_tags)

    data['game'].num_nodes = num_games
    data['game'].x = game_features

    if num_tags > 0:
        data['tag'].num_nodes = num_tags
        data['tag'].x = tag_features
    if num_devs > 0:
        data['developer'].num_nodes = num_devs
        data['developer'].x = dev_features

    # Game-Tag Edges
    if num_tags > 0 and 'steamspy_tag_list' in core_game_data_df.columns:
        game_tag_edge_sources = []
        game_tag_edge_targets = []
        for _, row in core_game_data_df.iterrows():
            game_node_id = game_id_map.get(row['steam_appid'])
            if game_node_id is not None and isinstance(row['steamspy_tag_list'], list):
                for tag_str in row['steamspy_tag_list']:
                    tag_node_id = tag_id_map.get(tag_str)
                    if tag_node_id is not None: # Ensure tag exists in map
                        game_tag_edge_sources.append(game_node_id)
                        game_tag_edge_targets.append(tag_node_id)

        if game_tag_edge_sources:
            data['game', 'has_tag', 'tag'].edge_index = torch.tensor(
                [game_tag_edge_sources, game_tag_edge_targets], dtype=torch.long
            )
            print(f"Created {len(game_tag_edge_sources)} game-has_tag-tag edges.")
        else:
            print("No game-tag edges were created (possibly no games with valid tags).")
    else:
        print("Skipping game-tag edges as 'steamspy_tag_list' is missing or no tags found.")

    # Game-Developer Edges
    if num_devs > 0 and 'developer' in core_game_data_df.columns:
        game_dev_edge_sources = []
        game_dev_edge_targets = []
        for _, row in core_game_data_df.iterrows():
            game_node_id = game_id_map.get(row['steam_appid'])
            developer_str = row['developer']
            if game_node_id is not None and pd.notna(developer_str): # Ensure developer is not NaN
                dev_node_id = dev_id_map.get(developer_str)
                if dev_node_id is not None: # Ensure developer exists in map
                    game_dev_edge_sources.append(game_node_id)
                    game_dev_edge_targets.append(dev_node_id)

        if game_dev_edge_sources:
            data['game', 'developed_by', 'developer'].edge_index = torch.tensor(
                [game_dev_edge_sources, game_dev_edge_targets], dtype=torch.long
            )
            print(f"Created {len(game_dev_edge_sources)} game-developed_by-developer edges.")
        else:
            print("No game-developer edges were created (possibly no games with valid developers).")
    else:
        print("Skipping game-developer edges: 'developer' column missing, or no developers found.")

    # Game-Game Similarity Edges (Jaccard Similarity based on 'steamspy_tag_list')
    if 'steamspy_tag_list' in core_game_data_df.columns and num_games > 1:
        print(f"\nCalculating game-game similarity (Jaccard on tags, threshold > {similarity_threshold})...")
        game_sim_sources = []
        game_sim_targets = []

        # Pre-calculate tag sets for each game for efficiency
        game_tag_sets = {}
        for appid, game_node_id in game_id_map.items():
            row = core_game_data_df.loc[core_game_data_df['steam_appid'] == appid]
            if not row.empty and isinstance(row.iloc[0]['steamspy_tag_list'], list):
                game_tag_sets[game_node_id] = set(row.iloc[0]['steamspy_tag_list'])
            else:
                game_tag_sets[game_node_id] = set()

        game_node_ids = list(game_id_map.values()) # Ensure we iterate over mapped IDs

        for i in range(num_games):
            for j in range(i + 1, num_games): # Avoid self-loops and duplicate pairs
                game1_node_id = game_node_ids[i]
                game2_node_id = game_node_ids[j]

                set1 = game_tag_sets.get(game1_node_id, set())
                set2 = game_tag_sets.get(game2_node_id, set())

                sim = jaccard_similarity(set1, set2)

                if sim > similarity_threshold:
                    game_sim_sources.extend([game1_node_id, game2_node_id]) # Add edges in both directions
                    game_sim_targets.extend([game2_node_id, game1_node_id])

        if game_sim_sources:
            data['game', 'similar_to', 'game'].edge_index = torch.tensor(
                [game_sim_sources, game_sim_targets], dtype=torch.long
            )
            print(f"Created {len(game_sim_sources)} game-similar_to-game edges (symmetric).")
        else:
            print("No game-similarity edges created above the threshold.")
    else:
        print("Skipping game-similarity edges: 'steamspy_tag_list' missing, or not enough games.")

    print("\n--- HeteroData Object (Post Feature and Similarity Edge Creation) ---")
    print(data)
    print("-------------------------------------------------------------------\n")
    return data, game_id_map, tag_id_map, dev_id_map, user_id_map


if __name__ == '__main__':
    # Configuration for the main execution
    SIMULATE_USERS_FLAG = True
    NUM_SYNTHETIC_USERS_FOR_TEST = 500 # Can be adjusted

    print("--- Starting Data Loading and Preprocessing for Graph Construction ---")
    steam_df = load_steam_data()
    description_df = load_steam_description_data()
    tags_df = load_steamspy_tag_data()

    core_game_data_df = preprocess_and_merge_data(steam_df, description_df, tags_df)

    if core_game_data_df.empty:
        print("Core game data is empty. Cannot build graph.")
    else:
        print(f"\n--- Building Heterogeneous Graph (Simulate Users: {SIMULATE_USERS_FLAG}) ---")
        graph_data, game_map, tag_map, dev_map, user_map = build_heterogeneous_graph(
            core_game_data_df,
            simulate_users=SIMULATE_USERS_FLAG,
            num_synthetic_users=NUM_SYNTHETIC_USERS_FOR_TEST
        )
        print("Successfully built heterogeneous graph and mappings.")

        print("\n--- Graph Node Summary ---")
        for node_type in graph_data.node_types:
            print(f"Node type '{node_type}': {graph_data[node_type].num_nodes} nodes.")
            if hasattr(graph_data[node_type], 'x') and graph_data[node_type].x is not None:
                print(f"  Features shape: {graph_data[node_type].x.shape}")
            else:
                print("  No features assigned.")

        print("\n--- Graph Edge Summary ---")
        for edge_type in graph_data.edge_types:
            print(f"Edge type {edge_type}: {graph_data[edge_type].num_edges} edges.")
            if hasattr(graph_data[edge_type], 'edge_index') and graph_data[edge_type].edge_index is not None:
                 print(f"  Edge index shape: {graph_data[edge_type].edge_index.shape}")

    print("\n--- Graph Builder Script Execution Finished ---")
