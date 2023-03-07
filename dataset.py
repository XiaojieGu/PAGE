import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaTokenizer


class BaseDataset(Dataset):
    def __init__(self, model_size='base', phase='train'):
        super(BaseDataset, self).__init__()
        self.emotion_dict = {'happiness': 0, 'neutral': 1, 'anger': 2, 'sadness': 3, 'fear': 4, 'surprise': 5, 'disgust': 6}
        self.act_mapping = {'inform': 0, 'question': 1, 'directive': 2, 'commissive': 3}
        self.speaker_dict = {'A': 0, 'B': 1}
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-'+model_size)
        data_path = './dataset/dailydialog_'+phase+'.pkl'
        processed_data_path = './dataset/dailydialog_processed_'+phase+'.pkl'

        if os.path.exists(processed_data_path):
            data = pickle.load(open(processed_data_path, 'rb'), encoding='latin1')
            self.utterance = data[0]
            self.token_ids = data[1]
            self.attention_mask = data[2]
            self.emotion = data[3]
            self.adj_index = data[4]
            self.evidence = data[5]
            self.mask = data[6]
            self.conv_len = data[7]
            self.act_label = data[8]
        else:
            data = pickle.load(open(data_path, 'rb'), encoding='latin1')
            self.utterance = []
            self.token_ids = []
            self.attention_mask = []
            self.emotion = []
            self.adj_index = []
            self.evidence = []
            self.mask = []
            self.conv_len = []
            self.act_label = []
            for d in data:
                utter = []
                tk_id = []
                at_mk = []
                emo = []
                spk = []
                evi = []
                act_lbl = []
                self.conv_len.append(len(d))
                for utt in d:
                    u = utt['utterance']
                    encoded_output = self.tokenizer(u)
                    tid = encoded_output.input_ids
                    atm = encoded_output.attention_mask
                    tk_id.append(torch.tensor(tid, dtype=torch.long))
                    at_mk.append(torch.tensor(atm))
                    e = utt['emotion']
                    s = utt['speaker']
                    i = utt['id']
                    a = utt['act']
                    if 'evidence' in utt:
                        ev = utt['evidence']
                    else:
                        ev = []
                    utter.append(u)
                    emo.append(self.emotion_dict[e])
                    spk.append(self.speaker_dict[s])
                    ev_vic = [0] * len(d)
                    for i in ev:
                        if i != 'b':
                            ev_vic[i - 1] = 1
                    evi.append(ev_vic)
                    act_lbl.append(self.act_mapping[a])

                evi = torch.tensor(evi, dtype=torch.long)
                msk = torch.ones_like(evi, dtype=torch.long).tril(0)
                spk = torch.tensor(spk)
                act_lbl = torch.tensor(act_lbl)
                same_spk = spk.unsqueeze(1) == spk.unsqueeze(0)
                other_spk = same_spk.eq(False).long().tril(0)
                same_spk = same_spk.long().tril(0)
                spker = torch.stack([same_spk, other_spk], dim=0)
                tk_id = pad_sequence(tk_id, batch_first=True, padding_value=1)
                at_mk = pad_sequence(at_mk, batch_first=True, padding_value=0)
                emo = torch.tensor(emo, dtype=torch.long)
                
                self.utterance.append(utter)
                self.token_ids.append(tk_id)
                self.attention_mask.append(at_mk)
                self.emotion.append(emo)
                self.evidence.append(evi)
                self.mask.append(msk)
                self.adj_index.append(spker)
                self.act_label.append(act_lbl)
            to_be_saved_data = [self.utterance, self.token_ids, self.attention_mask, self.emotion,
                                self.adj_index, self.evidence, self.mask, self.conv_len, self.act_label]
            pickle.dump(to_be_saved_data, open(processed_data_path, 'wb'))

    def __getitem__(self, item):
        utter = self.utterance[item]
        token_id = self.token_ids[item]
        att_mask = self.attention_mask[item]
        label = self.emotion[item]
        adj_idx = self.adj_index[item]
        evid = self.evidence[item]
        msk = self.mask[item]
        clen = self.conv_len[item]
        act = self.act_label[item]

        return utter, token_id, att_mask, label, adj_idx, evid, msk, clen, act

    def __len__(self):
        return len(self.emotion)



def collate_fn(data):
    token_ids = []
    token_ids_1 = []
    attention_mask = []
    attention_mask_1 = []
    label = []
    adj_index = []
    ece_pair = []
    mask = []
    clen = []
    act_label = []
    for i, d in enumerate(data):
        if i == 0:
            token_ids = d[1]
            attention_mask = d[2]
        else:
            d_tids = d[1]
            max_len = max(token_ids.shape[1], d_tids.shape[1])
            if token_ids.shape[1] < max_len:
                token_ids = torch.cat([token_ids, torch.ones(token_ids.shape[0], max_len-token_ids.shape[1], dtype=torch.long)], dim=1)
            if d_tids.shape[1] < max_len:
                d_tids = torch.cat([d_tids, torch.ones(d_tids.shape[0], max_len - d_tids.shape[1], dtype=torch.long)], dim=1)
            token_ids = torch.cat([token_ids, d_tids], dim=0)

            a_msk = d[2]
            max_len = max(attention_mask.shape[1], a_msk.shape[1])
            if attention_mask.shape[1] < max_len:
                attention_mask = torch.cat([attention_mask, torch.zeros(attention_mask.shape[0], max_len-attention_mask.shape[1], dtype=torch.long)], dim=1)
            if a_msk.shape[1] < max_len:
                a_msk = torch.cat([a_msk, torch.zeros(a_msk.shape[0], max_len-a_msk.shape[1], dtype=torch.long)], dim=1)
            attention_mask = torch.cat([attention_mask, a_msk], dim=0)
        label.append(d[3])

        token_ids_1.append(d[1])
        attention_mask_1.append(d[2])
        adj_index.append(d[4])
        ece_pair.append(d[5])
        mask.append(d[6])
        clen.append(d[7])
        act_label.append(d[8])

    label = pad_sequence(label, batch_first=True, padding_value=-1)
    act_label = pad_sequence(act_label, batch_first=True, padding_value=-1)
    max_len = max(clen)
    mask = [torch.cat([torch.cat([m, torch.zeros(max_len-m.shape[0], m.shape[1])], dim=0), torch.zeros(max_len, max_len-m.shape[1])], dim=1) for m in mask]
    mask = torch.stack(mask, dim=0)
    ece_pair = [torch.cat([torch.cat([ep, torch.zeros(max_len-ep.shape[0], ep.shape[1])], dim=0), torch.zeros(max_len, max_len-ep.shape[1])], dim=1) for ep in ece_pair]
    ece_pair = torch.stack(ece_pair, dim=0)
    adj_index = [torch.cat([torch.cat([a, torch.zeros(2, max_len-a.shape[1], a.shape[2])], dim=1), torch.zeros(2, max_len, max_len-a.shape[2])], dim=2) for a in adj_index]

    adj_index = torch.stack(adj_index, dim=0)

    return token_ids_1, attention_mask_1, clen, mask, adj_index, label, ece_pair, act_label
    

        


def get_dataloaders(model_size, batch_size, valid_shuffle):

    train_set = BaseDataset(model_size, 'train')
    dev_set = BaseDataset(model_size, 'dev')
    dd_set = BaseDataset(model_size, 'DD')
    ie_set = BaseDataset(model_size,'IE')

    train_loader = DataLoader(train_set, batch_size, True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_set, batch_size, valid_shuffle, collate_fn=collate_fn)
    dd_loader = DataLoader(dd_set, batch_size, valid_shuffle, collate_fn=collate_fn)
    ie_loader = DataLoader(ie_set, batch_size, valid_shuffle, collate_fn=collate_fn)

    return train_loader, dev_loader, dd_loader, ie_loader

