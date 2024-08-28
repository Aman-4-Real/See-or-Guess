'''
Author: Aman
Date: 2023-10-30 20:47:02
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2024-08-27 16:34:52
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class TE_loss(nn.Module):
    '''
    https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch
    '''
    def __init__(self, label_smoothing: float = 0.1, reduction="mean", weight=None):
        super(TE_loss, self).__init__()
        self.smoothing = label_smoothing
        self.reduction = reduction
        self.weight    = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
            if self.reduction == 'sum' else loss

    def forward(self,
                preds,
                gen_cap_preds,
                masked_item_lens,
                entities_ids,
                prefixes_lens,
                prompt_length):
        assert 0 <= self.smoothing < 1, "smoothing should be in [0,1)"

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        # masked_item_lens: [2, 1, 3, 5, 2, 5, 2, 3] => [0, 0, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 6, 6, 7, 7, 7]
        repeated = torch.repeat_interleave(torch.arange(len(masked_item_lens)), \
                                            torch.tensor(masked_item_lens)).to(preds.device)
        log_preds = torch.index_select(log_preds, 0, repeated)
        
        log_preds_gen = F.log_softmax(gen_cap_preds, dim=-1)

        total_loss = []
        for i in range(len(entities_ids)):
            entity_ids_i = entities_ids[i]
            prefix_len = prefixes_lens[i]
            sample_loss = torch.tensor(0.0).to(preds.device)
            for j, entity_id in enumerate(entity_ids_i):
                if prompt_length + prefix_len - 1 + j >= log_preds.shape[1]:
                    break
                pred_kw_loss = log_preds[i, prompt_length + prefix_len - 1 + j, entity_id]
                gen_kw_loss = log_preds_gen[i, :, entity_id].mean()
                each_loss = pred_kw_loss - gen_kw_loss
                sample_loss += each_loss
            total_loss.append(sample_loss / len(entity_ids_i))
        total_loss = torch.stack(total_loss)
        loss = self.reduce_loss(-total_loss)

        return loss




class NDE_loss(nn.Module):
    '''
    https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch
    '''
    def __init__(self, label_smoothing: float = 0.1, reduction="mean", weight=None):
        super(NDE_loss, self).__init__()
        self.smoothing = label_smoothing
        self.reduction = reduction
        self.weight    = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
            if self.reduction == 'sum' else loss

    def forward(self,
                gen_ori_img_preds,
                gen_cap_preds,
                masked_item_lens,
                entities_ids,
                prefixes_lens,
                prompt_length):
        assert 0 <= self.smoothing < 1, "smoothing should be in [0,1)"

        if self.weight is not None:
            self.weight = self.weight.to(gen_ori_img_preds.device)

        n = gen_ori_img_preds.size(-1)
        log_gen_ori_img_preds = F.log_softmax(gen_ori_img_preds, dim=-1)
        # masked_item_lens: [2, 1, 3, 5, 2, 5, 2, 3] => [0, 0, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 6, 6, 7, 7, 7]
        repeated = torch.repeat_interleave(torch.arange(len(masked_item_lens)), \
                                            torch.tensor(masked_item_lens)).to(gen_ori_img_preds.device)
        log_preds = torch.index_select(log_gen_ori_img_preds, 0, repeated)
        
        log_preds_gen = F.log_softmax(gen_cap_preds, dim=-1)

        total_loss = []
        for i in range(len(entities_ids)):
            entity_ids_i = entities_ids[i]
            prefix_len = prefixes_lens[i]
            sample_loss = torch.tensor(0.0).to(log_preds.device)
            for j, entity_id in enumerate(entity_ids_i):
                gen_ori_img_kw_loss = log_preds[i, :, entity_id].mean()
                gen_kw_loss = log_preds_gen[i, :, entity_id].mean()
                each_loss = gen_ori_img_kw_loss - gen_kw_loss
                sample_loss += each_loss
            total_loss.append(sample_loss / len(entity_ids_i))

        total_loss = torch.stack(total_loss)
        loss = self.reduce_loss(-total_loss)

        return loss

