import json

class DataLoader:
    def __init__(self, json_path="../data/cui_relations_output_llama_point_four.json"):
        with open(json_path, "r", encoding="utf-8") as f:
            self.training_data = json.load(f)

    def get_training_data(self):
        return self.training_data