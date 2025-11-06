import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("✅ ingredient_recommender.py loaded successfully!")


# ---------------- LOAD INGREDIENT DATA ----------------
def load_ingredients():
    INGREDIENT_PATH = "data/ingredients.csv"  # adjust if needed
    ingredients_df = pd.read_csv(INGREDIENT_PATH, encoding="utf-8")

    # Combine relevant fields
    ingredients_df["combined_text"] = (
        ingredients_df["short_description"].fillna('') + " " +
        ingredients_df["what_is_it"].fillna('') + " " +
        ingredients_df["what_does_it_do"].fillna('') + " " +
        ingredients_df["who_is_it_good_for"].fillna('')
    )

    return ingredients_df


# ---------------- LOAD MODEL ONCE ----------------
_encoder = None
_ingredient_embeddings = None
_ingredients_df = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer('all-MiniLM-L6-v2')
    return _encoder


def _get_embeddings():
    global _ingredient_embeddings, _ingredients_df
    if _ingredient_embeddings is None or _ingredients_df is None:
        _ingredients_df = load_ingredients()
        encoder = _get_encoder()
        _ingredient_embeddings = encoder.encode(
            _ingredients_df["combined_text"].tolist(),
            show_progress_bar=False
        )
    return _ingredient_embeddings, _ingredients_df


# ---------------- RECOMMEND FUNCTION ----------------
def recommend_ingredients(skin_issues, top_n=5):
    """
    Recommend skincare ingredients based on detected skin issues.
    Example:
        recommend_ingredients(['acne', 'dry skin'])
    """
    encoder = _get_encoder()
    ingredient_embeddings, ingredients_df = _get_embeddings()

    query_text = " ".join(skin_issues)
    query_embed = encoder.encode([query_text])
    similarities = cosine_similarity(query_embed, ingredient_embeddings)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]

    recommendations = ingredients_df.iloc[top_indices][
        ["name", "short_description", "who_is_it_good_for"]
    ]
    return recommendations.to_dict(orient="records")
