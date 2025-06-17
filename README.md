# Game Recommendation System using Graph Neural Networks

*<p align="center">A placeholder for a screenshot of the application.</p>*

## Overview

This project is a sophisticated game recommendation system that leverages the power of Graph Neural Networks (GNNs) to provide users with personalized game suggestions. The system is delivered through a sleek, modern web interface built with Flask and interactive JavaScript.

The core of the recommendation engine is a GNN model trained on game data to learn complex relationships and similarities between games. This allows for highly accurate recommendations based on a user's input of 1-5 games they enjoy.

## How It Works: The GNN Approach

This project moves beyond traditional recommendation algorithms by modeling the relationships between games as a graph. This allows us to capture not just what games are similar based on user ratings, but also the underlying structural connections between them.

### What is a Graph Neural Network (GNN)?

In our context, a Graph Neural Network (GNN) treats the entire library of games as a network, or "graph."

-   **Nodes**: Each game in the dataset is a *node* in the graph.
-   **Edges**: Connections, or *edges*, between nodes can represent various relationships, such as shared genres, similar player bases, or developers.

The GNN learns by "message passing," where each node aggregates information from its neighbors. After several rounds of message passing, each node (game) develops a rich numerical representation called an **embedding**. This embedding is a dense vector that encodes the game's characteristics and its position within the entire game ecosystem.

### Why are GNNs Better for Recommendations?

Traditional machine learning models, like collaborative filtering, often struggle with:
-   **Capturing Complex Relationships**: They primarily rely on user-item interaction matrices and may miss subtle, transitive relationships (e.g., users who like game A also like game C, because both are related to game B).
-   **Data Sparsity**: They perform poorly when interaction data is sparse.

GNNs overcome these limitations by:
1.  **Learning from Graph Structure**: They explicitly use the connections between games to inform the embeddings, capturing deeper, more complex patterns.
2.  **Creating Richer Embeddings**: The resulting game embeddings are more powerful because they encode both the game's own features (content) and its relationships with other games (context).
3.  **Inductive Capabilities**: GNNs can generalize to new, unseen nodes (games) more effectively than many traditional methods, helping with the "cold start" problem.

### Our Project's Workflow

The recommendation process follows these steps:

1.  **Offline Model Training (in `v1/` and `v2/` notebooks)**:
    *   A graph is constructed from the `steam.csv` dataset.
    *   A GNN model is trained on this graph. The goal is to learn a function that produces a high-quality embedding for each game.
    *   The final trained model (`gcn_game_model.pt`) and the resulting game embeddings (`gcn_game_embeddings.pt`) are saved.

2.  **Online Recommendation (in the Web App)**:
    *   The Flask application loads the pre-computed `gcn_game_embeddings.pt`.
    *   When a user selects one or more games, the system retrieves the embeddings for these games.
    *   It calculates the *average embedding* of the selected games. This average vector represents the user's current taste profile.
    *   **Cosine Similarity** is then used to measure the "distance" between this taste profile vector and every other game embedding in the dataset.
    *   The games with the highest similarity scores (i.e., the "closest" vectors) are selected, filtered to remove the input games, and presented to the user as recommendations.

## Features

- **AI-Powered Recommendations**: Utilizes a pre-trained GNN model (`gcn_game_model.pt`) and game embeddings (`gcn_game_embeddings.pt`) to generate recommendations.
- **Interactive UI**: A user-friendly web interface for searching and selecting games.
- **Autocomplete Search**: An intelligent search bar that provides real-time suggestions as the user types, ensuring accurate game selection from the dataset.
- **Multi-Game Input**: Users can select 1 to 5 games to get more nuanced and personalized recommendations.
- **Modern & Responsive Design**: A clean, visually appealing interface with smooth animations and transitions, built to work seamlessly on both desktop and mobile devices.
- **Modular Codebase**: The project is structured with a clear separation between the backend (Flask), frontend (HTML/CSS/JS), and the machine learning model.

## Project Structure

The repository is organized to separate the different stages of the project:

```
.gcn-game-recommender/
├── data/                     # Contains the raw CSV data for games
│   └── steam.csv
├── templates/                # Flask templates for the web UI
│   └── index.html
├── v1/                       # Version 1 of the model development (.ipynb)
│   └── ...
├── v2/                       # Version 2 (final) of the model development (.ipynb)
│   └── ...
├── .gitignore                # Specifies files to be ignored by Git
├── app.py                    # Main Flask application file
├── gcn_game_embeddings.pt    # Pre-computed game embeddings
├── gcn_game_model.pt         # Pre-trained GNN model state
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

- **`v1/` & `v2/`**: These directories contain the Jupyter Notebooks that document the development and evolution of the GNN model. `v1` represents the initial approach, while `v2` contains the final, polished version of the model training process.
- **`app.py`**: The Flask backend that serves the web UI and handles API requests for search and recommendations.
- **`templates/index.html`**: The single-page frontend for the application.
- **`.pt` files**: The core of the recommendation engine. These are the outputs of the model training process from the notebooks.

## Technology Stack

- **Backend**: Python, Flask, Flask-CORS
- **Machine Learning**: PyTorch, Pandas, NumPy
- **Frontend**: HTML5, CSS3, JavaScript
- **UI/UX**: jQuery, Select2 for the search bar, Animate.css for effects

## Setup and Installation

Follow these steps to get the application running on your local machine.

### 1. Prerequisites

- Python 3.8+
- `pip` package manager

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd gcn-game-recommender
```

### 3. Set Up a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

- **On macOS/Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- **On Windows**:
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```

### 4. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Running the Application

Once the setup is complete, you can start the web server.

1.  Run the `app.py` script:
    ```bash
    python app.py
    ```

2.  The application will start a local development server. Open your web browser and navigate to:
    [http://127.0.0.1:5000](http://127.0.0.1:5000)

## How to Use the Web Interface

1.  **Search for Games**: Start typing the name of a game you like in the search box.
2.  **Select Games**: Choose between 1 and 5 games from the search results. Your selections will appear as tags.
3.  **Get Recommendations**: Click the "Get Recommendations" button.
4.  **View Results**: The application will display a list of recommended games based on your selections.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
