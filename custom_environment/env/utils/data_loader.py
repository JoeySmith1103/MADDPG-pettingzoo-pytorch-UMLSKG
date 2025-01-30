import json
import os

class DataLoader:
    def __init__(self, json_path=None):
        if json_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, "..", "data", "cui_relations_output_llama_point_four.json")
        
        with open(json_path, "r", encoding="utf-8") as f:
            self.training_data = json.load(f)

    def get_training_data(self):
        return self.training_data
    
    def get_all_llama_keyword_cuis(self):
        dic = {}
        data = {}
        for item in self.training_data:
            index = item.get("index")
            if index:
                if index not in dic:
                    dic[index] = item["query_keyword"]
                    llama_outputs = item["llama2_output"]
                    data[index] = set()
                    for output in llama_outputs.values():
                        cui = output.get("CUI")
                        if cui:
                            data[index].add(cui)
                else:
                    continue
        return {index: list(cuis) for index, cuis in data.items()}
