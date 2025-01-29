from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import numpy as np

class GraphHandler:
    def __init__(self):
        self.driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "admin"))
        self.tokenizer = AutoTokenizer.from_pretrained("JoeySmith1103/sapbert_2023ABFull")
        self.model = AutoModel.from_pretrained("JoeySmith1103/sapbert_2023ABFull").cuda()

    def close(self):
        self.driver.close()

    def get_all_concept_nodes(self):
        query = "MATCH (c:Concept) RETURN c.CUI AS node_id"
        with self.driver.session() as session:
            result = session.run(query)
            return [record["node_id"] for record in result]

    def find_one_hop_neighbors(self, node_id):
        query = """
        MATCH (c:Concept {CUI: $cui})-[:PAR|:CHD|:RN|:AQ|:SY*1]->(neighbor:Concept)
        RETURN neighbor.CUI AS neighbor_cui, neighbor.name AS neighbor_name
        """
        with self.driver.session() as session:
            result = session.run(query, cui=node_id)
            return [{"neighbor_cui": record["neighbor_cui"], "neighbor_name": record["neighbor_name"]} for record in result]

    def get_embeddings(self, names, batch_size=128):
        all_embs = []
        for i in range(0, len(names), batch_size):
            toks = self.tokenizer.batch_encode_plus(names[i:i + batch_size], padding="max_length",
                                                    max_length=25, truncation=True, return_tensors="pt")
            toks_cuda = {k: v.cuda() for k, v in toks.items()}
            cls_rep = self.model(**toks_cuda)[0][:, 0, :]
            all_embs.append(cls_rep.cpu().detach().numpy())
        return np.concatenate(all_embs, axis=0)

    def calculate_average_similarity(self, node_id):
        neighbors = self.find_one_hop_neighbors(node_id)
        if not neighbors:
            return {"num_neighbors": 0, "avg_similarity": 0.0, "variance_similarity": 0.0}

        neighbor_names = [record['neighbor_name'] for record in neighbors]
        embeddings = self.get_embeddings(neighbor_names)
        similarity_matrix = cosine_similarity(embeddings)
        similarities = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]

        return {
            "num_neighbors": len(neighbor_names),
            "avg_similarity": np.mean(similarities),
            "variance_similarity": np.var(similarities),
        }