import torch
import math

def compute_acc(GroundTruth, predictedIndices, topN):
    precision = []
    recall = []
    NDCG = []
    MRR = []

    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0 / math.log2(j + 2)
                        if mrrFlag:
                            userMRR = 1.0 / (j + 1.0)
                            mrrFlag = False
                        userHit += 1

                    if idcgCount > 0:
                        idcg += 1.0 / math.log2(j + 2)
                        idcgCount = idcgCount - 1

                if idcg != 0:
                    ndcg += dcg / idcg

                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])
                sumForNdcg += ndcg
                sumForMRR += userMRR

        precision.append(sumForPrecision / len(predictedIndices))
        recall.append(sumForRecall / len(predictedIndices))
        NDCG.append(sumForNdcg / len(predictedIndices))
        MRR.append(sumForMRR / len(predictedIndices))

    return precision, recall, NDCG, MRR

def test_all_users(model, item_num, test_data_pos, user_pos, top_k, device='cuda'):
    predictedIndices = []
    GroundTruth = []
    for u in test_data_pos:
        batch_user = torch.tensor([u] * item_num, dtype=torch.long, device=device)
        batch_item = torch.tensor(
            [i for i in range(item_num)], dtype=torch.long, device=device
        )

        predictions = model(batch_user, batch_item)
        test_data_mask = torch.zeros(item_num, device=device)
        if u in user_pos:
            test_data_mask[user_pos[u]] = -9999
        predictions = predictions + test_data_mask
        _, indices = torch.topk(predictions, top_k[-1])
        indices = indices.cpu().numpy().tolist()
        predictedIndices.append(indices)
        GroundTruth.append(test_data_pos[u])
    precision, recall, NDCG, MRR = compute_acc(GroundTruth, predictedIndices, top_k)
    return precision, recall, NDCG, MRR

def test_all_users_cdae(model, item_num, test_data_pos, user_pos, top_k, observed_mat, device='cuda'):
    predictedIndices = []
    GroundTruth = []
    for u in test_data_pos:
        user = torch.tensor(u, dtype=torch.long, device=device)
        item_vec = torch.tensor(observed_mat.getrow(u).toarray()[0], device=device, dtype=torch.float32)

        predictions = model(user, item_vec)
        test_data_mask = torch.zeros(item_num, device=device)
        if u in user_pos:
            test_data_mask[user_pos[u]] = -9999
        predictions = predictions + test_data_mask
        _, indices = torch.topk(predictions, top_k[-1])
        indices = indices.cpu().numpy().tolist()
        predictedIndices.append(indices)
        GroundTruth.append(test_data_pos[u])
    precision, recall, NDCG, MRR = compute_acc(GroundTruth, predictedIndices, top_k)
    return precision, recall, NDCG, MRR