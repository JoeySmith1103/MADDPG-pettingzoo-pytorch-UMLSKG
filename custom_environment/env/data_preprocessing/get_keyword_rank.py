from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel
import json
from collections import OrderedDict

class KeywordRanker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("JoeySmith1103/sapbert_2023ABFull")
        self.model = AutoModel.from_pretrained("JoeySmith1103/sapbert_2023ABFull").cuda()
        self.true_label_data = self.get_true_label_data("records.json", 100)
        self.candidate_label_data = self.get_candidate_label_data("records_with_cui_tui_llama_2.json", 100)

    def get_true_label_data(self, data_name, nums):
        with open(f'/mnt/data/home/joeysmith/maddpg-pettingzoo-pytorch/custom_environment/env/data/{data_name}', 'r') as file:
            origin_data = json.load(file)
        true_label_array = []
        for data in origin_data[:nums]:
            true_label_array.append(list(set(data['true_label_from_allergies_and_chief_complaint'])))
        return true_label_array
    
    def get_candidate_label_data(self, data_name, nums):
        with open(f'/mnt/data/home/joeysmith/maddpg-pettingzoo-pytorch/custom_environment/env/data/{data_name}', 'r') as file:
            curr_data = json.load(file)

        candidate_label_array = [[] for _ in range(nums)]
        for i, item in enumerate(curr_data[:nums]):
            temp = []
            for element in item['cui_tui']:  # 使用 item 而不是 curr_data[i]
                temp.append(element["entity"])
            candidate_label_array[i] = list(set(temp))

        return candidate_label_array
    
    def get_embeddings(self, names, tokenizer, model, batch_size=128):
        all_embs = []
        for i in tqdm(np.arange(0, len(names), batch_size)):
            toks = tokenizer.batch_encode_plus(names[i:i+batch_size],
                                            padding="max_length",
                                            max_length=25,
                                            truncation=True,
                                            return_tensors="pt")
            toks_cuda = {k: v.cuda() for k, v in toks.items()}
            cls_rep = model(**toks_cuda)[0][:, 0, :]  # Use CLS representation as the embedding
            all_embs.append(cls_rep.cpu().detach().numpy())
        all_embs = np.concatenate(all_embs, axis=0)
        return all_embs


    # Function to compute cosine similarity and filter entities
    def ranking(self, entities, embeddings):
        rank = {}
        for entity in entities:
            toks = self.tokenizer.encode(entity, return_tensors="pt").cuda()
            cls_rep = self.model(toks)[0][:, 0, :].cpu().detach().numpy()
            sim_scores = cosine_similarity(cls_rep, embeddings)
            rank[entity] = np.mean(sim_scores)

        # 依照相似度排序 (從高到低)
        sorted_rank = OrderedDict(sorted(rank.items(), key=lambda x: x[1], reverse=True))
        return sorted_rank
    
test = KeywordRanker()
print(test.true_label_data[0])
print(test.candidate_label_data[0])
true_label_embeddings = [test.get_embeddings(true_labels, test.tokenizer, test.model) for true_labels in test.true_label_data]
print(test.ranking(test.candidate_label_data[0], true_label_embeddings[0]))