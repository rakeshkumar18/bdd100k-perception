from src.model_registry.model_registry import ModelRegistry

registry = ModelRegistry()

print(registry.get_best_model_path())
