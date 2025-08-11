import os
import pandas as pd
import numpy as np
import pickle
from scipy.sparse.linalg import svds

DATA_PATH = "data"
MODELS_PATH = "models"

BOOKS_FILE = os.path.join(DATA_PATH, "books_with_keywords.csv")
USERS_FILE = os.path.join(DATA_PATH, "cleaned_users.csv")
EXPLICIT_MATRIX_FILE = os.path.join(DATA_PATH, "explicit_interactions.csv")
SIM_MATRIX_FILE = os.path.join(MODELS_PATH, "cosine_similarity.pkl")

def load_resources():
    """Load and prepare all required resources."""
    # Load dataframes
    books_df = pd.read_csv(BOOKS_FILE)
    users_df = pd.read_csv(USERS_FILE)
    ratings_matrix = pd.read_csv(EXPLICIT_MATRIX_FILE, index_col=0)
    
    # Keep the original numeric index for similarity matrix lookups
    books_df['numeric_index'] = books_df.index
    
    # Set ISBN as index for ratings matrix operations
    books_df.set_index('ISBN', inplace=True)
    
    # Load similarity matrix
    with open(SIM_MATRIX_FILE, "rb") as f:
        sim_matrix = pickle.load(f)
    
    return books_df, users_df, ratings_matrix, sim_matrix

def get_user_preferences(user_id, ratings_matrix):
    if user_id in ratings_matrix.index:
        return ratings_matrix.loc[user_id]
    return None

def calculate_user_similarity(target_preferences, ratings_matrix):
    # Calculate cosine similarity between target user and all other users
    user_similarities = ratings_matrix.dot(target_preferences) / (
        np.sqrt(np.sum(ratings_matrix ** 2, axis=1)) * 
        np.sqrt(np.sum(target_preferences ** 2))
    )
    return user_similarities

def recommend_books(title=None, user_id=None, books_df=None, users_df=None, 
                   ratings_matrix=None, sim_matrix=None, top_n=5, blend_factor=0.5):
    """Recommend books based on content similarity and/or user preferences"""
    if not (title or user_id):
        raise ValueError("Must provide either title or user_id")
        
    content_scores = None
    collab_scores = None
    
    # Get content-based recommendations if title provided
    if title:
        title_lower = title.lower()
        matches = books_df[books_df["Book-Title"].str.lower() == title_lower]
        if not matches.empty:
            # Use numeric_index for similarity matrix lookup
            numeric_idx = matches['numeric_index'].iloc[0]
            content_scores = pd.Series(sim_matrix[numeric_idx], 
                                     index=books_df.index)
    
    # Get collaborative filtering recommendations if user_id provided
    if user_id and user_id in ratings_matrix.index:
        user_preferences = get_user_preferences(user_id, ratings_matrix)
        if user_preferences is not None:
            # Get similar users
            user_similarities = calculate_user_similarity(user_preferences, ratings_matrix)
            
            # Weight ratings by user similarity
            weighted_ratings = ratings_matrix.mul(user_similarities, axis=0)
            collab_scores = weighted_ratings.mean()
            
            # Ensure indices match books_df
            collab_scores = collab_scores[collab_scores.index.isin(books_df.index)]
    
    # Blend scores if both methods available
    if content_scores is not None and collab_scores is not None:
        # Ensure we only use indices that exist in both
        common_indices = content_scores.index.intersection(collab_scores.index)
        content_scores = content_scores[common_indices]
        collab_scores = collab_scores[common_indices]
        final_scores = (1 - blend_factor) * content_scores + blend_factor * collab_scores
    elif content_scores is not None:
        final_scores = content_scores
    elif collab_scores is not None:
        final_scores = collab_scores
    else:
        print("[WARN] Could not generate recommendations")
        return pd.DataFrame()
    
    # Get top recommendations
    try:
        top_indices = final_scores.nlargest(top_n + 1).index
        if title:  # Remove input book if title-based
            top_indices = [idx for idx in top_indices if idx != matches.index[0]][:top_n]
        
        recommendations = books_df.loc[top_indices]
        
        # Add score and rank
        recommendations["Score"] = final_scores[top_indices]
        recommendations["Rank"] = range(1, len(recommendations) + 1)
        
        return recommendations[["Rank", "Book-Title", "Book-Author", "Publisher", "Score"]]
    except KeyError as e:
        print(f"[ERROR] Failed to lookup books: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Load all resources
    books_df, users_df, ratings_matrix, sim_matrix = load_resources()
    
    # Example 1: Content-based recommendation
    print("\nContent-based recommendations for 'Harry Potter':")
    print(recommend_books(
        title="harry potter and the chamber of secrets",
        books_df=books_df,
        users_df=users_df,
        ratings_matrix=ratings_matrix,
        sim_matrix=sim_matrix,
        blend_factor=0.0  # Pure content-based
    ))
    
    # Example 2: User-based recommendation
    print("\nUser-based recommendations for user 1:")
    print(recommend_books(
        user_id=1,
        books_df=books_df,
        users_df=users_df,
        ratings_matrix=ratings_matrix,
        sim_matrix=sim_matrix,
        blend_factor=1.0  # Pure collaborative
    ))
    
    # Example 3: Hybrid recommendation
    print("\nHybrid recommendations for user 1 who likes Harry Potter:")
    print(recommend_books(
        title="harry potter and the chamber of secrets",
        user_id=1,
        books_df=books_df,
        users_df=users_df,
        ratings_matrix=ratings_matrix,
        sim_matrix=sim_matrix,
        blend_factor=0.5  # Equal blend
    ))
