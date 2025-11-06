# ai_routine_generator.py

from transformers import pipeline

# Load GPT-2 model from Hugging Face
# You can replace "gpt2" with "distilgpt2" (lighter) if memory is low
routine_generator = pipeline("text-generation", model="gpt2")

def generate_routine(issues, ingredients):
    """
    Generate a skincare routine based on detected issues and ingredients.

    Parameters:
    - issues (list): list of detected skin issues (e.g., ['acne', 'oily skin'])
    - ingredients (list of dict): recommended ingredients with names and benefits

    Returns:
    - str: generated skincare routine
    """
    
    # Create a clear, descriptive prompt for GPT-2
    ingredient_names = [i["name"] for i in ingredients]
    
    prompt = (
        f"You are a professional dermatologist. Create a skincare routine for someone who has "
        f"{', '.join(issues)}. "
        f"Recommended ingredients include: {', '.join(ingredient_names)}. "
        f"Suggest a simple daily routine (morning and night) using these ingredients. "
        f"Keep it friendly, concise, and practical."
    )

    # Generate text using GPT-2
    response = routine_generator(
        prompt,
        max_length=200,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.2,
        num_return_sequences=1
    )

    # Return only the generated routine text
    routine_text = response[0]["generated_text"]
    return routine_text
