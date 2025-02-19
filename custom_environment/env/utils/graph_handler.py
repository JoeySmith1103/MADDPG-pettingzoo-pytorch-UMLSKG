from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import numpy as np
import random

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.data_loader import DataLoader

class GraphHandler:
    def __init__(self):
        self.driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "admin"))
        self.tokenizer = AutoTokenizer.from_pretrained("JoeySmith1103/sapbert_2023ABFull")
        self.model = AutoModel.from_pretrained("JoeySmith1103/sapbert_2023ABFull").cuda()
        self.data_loader = DataLoader()
        self.ranked_data = self.data_loader.get_training_data()

    def close(self):
        self.driver.close()

    def get_top_k_cuis_from_randomly_chosen_one_data(self, k=10):
        """
        get top k cuis from one dataset
        """
        random_index = random.randint(0, len(self.ranked_data) - 1)
        selected_data = self.ranked_data[random_index]
        selected_ranking = selected_data["ranking"]
        top_k_cuis = []
        
        for entity, cui, _ , similarity in selected_ranking:
            if len(top_k_cuis) >= k:
                break 

            # check if cui has neighbors
            if self.find_one_hop_neighbors(cui):  
                top_k_cuis.append((cui, similarity))

        return top_k_cuis  # return top k cuis
    
    def find_one_hop_neighbors(self, node_id):
        query = """
        MATCH (c:Concept {CUI: $cui})-[:PAR|:CHD|:RN|:AQ|:SY*1]->(neighbor:Concept)
        RETURN neighbor.CUI AS neighbor_cui, neighbor.name AS neighbor_name
        """
        with self.driver.session() as session:
            result = session.run(query, cui=node_id)
            return [{"neighbor_cui": record["neighbor_cui"], "neighbor_name": record["neighbor_name"]} for record in result]

    def get_embeddings(self, names, tokenizer, model, batch_size=128):
        if not names:
            print("[Warning] Empty name list provided to get_embeddings(), returning empty array.")
            return np.array([])  # 直接回傳空陣列，避免錯誤
        
        all_embs = []
        for i in range(0, len(names), batch_size):
            toks = tokenizer.batch_encode_plus(names[i:i + batch_size],
                                            padding="max_length",
                                            max_length=25,
                                            truncation=True,
                                            return_tensors="pt")
            toks_cuda = {k: v.cuda() for k, v in toks.items()}
            cls_rep = model(**toks_cuda)[0][:, 0, :]
            all_embs.append(cls_rep.cpu().detach().numpy())

        all_embs = np.concatenate(all_embs, axis=0) if all_embs else np.array([])
        
        if all_embs.shape[0] == 0:
            print("[Warning] No embeddings computed, returning empty array.")

        return all_embs

    def calculate_average_similarity_among_neighbors(self, node):
        neighbors = self.find_one_hop_neighbors(node_id=node)

        if not neighbors:
            print(f"[Warning] Node {node} has no neighbors, returning 0 similarity.")
            return {"num_neighbors": 0, "avg_similarity": 0.0, "variance_similarity": 0.0}

        neighbor_names = [record['neighbor_name'] for record in neighbors]

        if len(neighbor_names) < 2:  # 至少要有 2 個 neighbor 才能計算相似度
            print(f"[Warning] Node {node} has less than 2 neighbors, returning 0 similarity.")
            return {"num_neighbors": len(neighbor_names), "avg_similarity": 0.0, "variance_similarity": 0.0}

        embeddings = self.get_embeddings(neighbor_names, self.tokenizer, self.model)

        if embeddings.shape[0] < 2:  # 確保 embeddings 數量足夠
            print(f"[Warning] Embeddings for Node {node} are insufficient, returning 0 similarity.")
            return {"num_neighbors": len(neighbor_names), "avg_similarity": 0.0, "variance_similarity": 0.0}

        similarity_matrix = cosine_similarity(embeddings)

        similarities = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]

        avg_similarity = np.mean(similarities) if similarities.size > 0 else 0.0
        variance_similarity = np.var(similarities) if similarities.size > 0 else 0.0

        return {
            "num_neighbors": len(neighbor_names),
            "avg_similarity": avg_similarity,
            "variance_similarity": variance_similarity,
        }
    
test = GraphHandler()
ge = test.get_top_k_cuis_from_randomly_chosen_one_data()
print(ge)
# length = []
# filter = [r for r in test.ranked_data[0]['ranking'] if r[3] > 0.4]
# print(len(filter))
# print(len(test.ranked_data[0]['ranking']))
# for entry in test.ranked_data:
#     # 過濾 similarity > 0.4 的 ranking
#     original_length = len(entry['ranking'])
#     filtered_ranking = [r for r in entry['ranking'] if r[3] > 0.4]
#     print(original_length, len(filtered_ranking))
#     length.append(len(filtered_ranking))

# # 計算平均長度和最小長度
# if length:  # 確保不會除以 0
#     print(f"similarity > 0.4 的平均 ranking 長度: {np.average(length):.2f}")
#     print(f"similarity > 0.4 的最小 ranking 長度: {min(length)}")
# else:
#     print("沒有符合 similarity > 0.4 的 ranking")
# print(test.ranked_data[99]['ranking'])
