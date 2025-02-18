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
        self.candidate_label_data = self.get_candidate_label_data("cui_relations_output.json", 100)

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

        seen_indices = set()
        candidate_label_array = []

        for item in curr_data:
            index = item["index"]
            if index in seen_indices:
                continue  # 跳過已處理過的 index
            seen_indices.add(index)

            llama_output = item.get("llama2_output", {})
            extracted_data = set()  # 用 set() 確保同一筆資料內不重複

            for entity, details in llama_output.items():
                cui = details.get("CUI", "Unknown_CUI")
                semantic_groups = details.get("semantic_group", [])

                # 取得所有 semantic group 的 "group" 值，並去重
                group_names = tuple(set(sg.get("group", "Unknown_Group") for sg in semantic_groups))  # **轉換為 tuple 避免 set() 出錯**

                extracted_data.add((entity, cui, group_names))  # make sure entity won't be duplicated

            candidate_label_array.append(list(extracted_data))  # **轉回 list 方便 JSON 存取**

            # 若已達到 nums 限制則停止
            if len(candidate_label_array) >= nums:
                break

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

    def ranking(self, entities_with_cui_group, embeddings):
        rank = {}

        for entity, cui, groups in entities_with_cui_group:
            toks = self.tokenizer.encode_plus(entity, return_tensors="pt", 
                                            padding="max_length", max_length=25, truncation=True)
            toks_cuda = {k: v.cuda() for k, v in toks.items()}
            cls_rep = self.model(**toks_cuda)[0][:, 0, :].cpu().detach().numpy()

            sim_scores = cosine_similarity(cls_rep, embeddings)
            rank[(entity, cui, tuple(groups))] = float(np.mean(sim_scores))  # 轉換為 Python float

        # sorted by similarity
        sorted_rank = sorted(rank.items(), key=lambda x: x[1], reverse=True)

        return [(entity, cui, list(groups), score) for (entity, cui, groups), score in sorted_rank]
    

# import json
# import os

# test = KeywordRanker()

# # 計算所有 true labels 的 embedding
# true_label_embeddings = [test.get_embeddings(true_labels, test.tokenizer, test.model) for true_labels in test.true_label_data]

# # 準備要存入 JSON 的資料
# results = []

# # 針對 100 筆資料執行 ranking
# for i in range(len(test.candidate_label_data)):
#     ranked_data = test.ranking(test.candidate_label_data[i], true_label_embeddings[i])
#     results.append({
#         "index": i,  # 加入 index 方便追蹤
#         "ranking": ranked_data
#     })

# # print(results[0])
# filtered_ranking = [r for r in results[0]['ranking'] if r[3] > 0.4]

# # **確保 custom_environment/env/data/ 目錄存在**
# output_dir = "custom_environment/env/data"
# os.makedirs(output_dir, exist_ok=True)  # 自動建立目錄（如果不存在）

# # **存檔到 custom_environment/env/data/ranking_results.json**
# output_path = os.path.join(output_dir, "ranking_results.json")
# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(results, f, ensure_ascii=False, indent=4)

# print(f"Ranking results saved to {output_path}")

## average length of keywords 89.69
## min length of keywords 54
## average length of keywords that similarity > 0.4 15
## min length of keywords that similarity > 0.4 1

# test = KeywordRanker()
# zerofour_length = []
# origin_length = []
# for entry in results:
#     # 過濾 similarity > 0.4 的 ranking
#     filtered_ranking = [r for r in entry['ranking'] if r[3] > 0.4]
#     zerofour_length.append(len(filtered_ranking))
#     origin_length.append(len(entry['ranking']))

# # 計算平均長度和最小長度
# if zerofour_length:  # 確保不會除以 0
#     print(f"原本的 ranking 長度: {np.average(origin_length):.2f}")
#     print(f"原本的 ranking 最小長度: {min(origin_length)}")
#     print(f"similarity > 0.4 的平均 ranking 長度: {np.average(zerofour_length):.2f}")
#     print(f"similarity > 0.4 的最小 ranking 長度: {min(zerofour_length)}")
# else:
#     print("沒有符合 similarity > 0.4 的 ranking")


