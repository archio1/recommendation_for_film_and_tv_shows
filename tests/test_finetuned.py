from src.models.finetuned import FinetunedRecommender

def test_finetuned_recommender():
    recommender = FinetunedRecommender(config_path="src/config/pretrained_config.yaml")
    
    prompt = "Recommend me a good sci-fi action movie."
    response = recommender.generate(prompt)
    
    assert isinstance(response, str), "Response should be a string"
    assert len(response) > 0, "Response should not be empty"

def test_comedy_recommendation():
    recommender = FinetunedRecommender(config_path="src/config/pretrained_config.yaml")
    
    prompt = "Порекомендуй комедії для перегляду?"
    response = recommender.generate(prompt, use_comedy_params=True)
    
    assert isinstance(response, str), "Response should be a string"
    assert len(response) > 0, "Response should not be empty"