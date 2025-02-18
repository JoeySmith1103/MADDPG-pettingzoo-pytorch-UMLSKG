import json
import os

class DataLoader:
    def __init__(self, json_path=None):
        # self.all_data_path = "cui_relations_output.json"
        # self.data_threshold_04_path = "cui_relations_output_llama_point_four.json"
        self.ranked_data_path = "ranking_results.json"
        if json_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # json_path = os.path.join(current_dir, "..", "data", "cui_relations_output_llama_point_four.json")
            # json_path = os.path.join(current_dir, "..", "data", self.all_data_path)
            json_path = os.path.join(current_dir, "..", "data", self.ranked_data_path)
        with open(json_path, "r") as f:
            self.ranked_data = json.load(f)

    def get_training_data(self):
        return self.ranked_data

