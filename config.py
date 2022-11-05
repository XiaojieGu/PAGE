import torch
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
TORCH_SEED = 123

DATA_DIR = '/home/DATA1/gxj/ECPE/original/data_reccon'
TRAIN_DIR = 'dailydialog_train.json'
VALID_DIR = 'dailydialog_valid.json'
TEST_DIR = 'dailydialog_test.json'



class Config(object):
    def __init__(self):
        self.split = 'split10'

        self.bert_cache_path = 'roberta-base'
        self.feat_dim = 768

        self.gnn_dim = '192'
        self.att_heads = '4'
        self.K = 12
        self.pos_emb_dim = 50
        self.pairwise_loss = False

        self.epochs = 15
        self.lr = 1e-5
        self.batch_size = 4
        self.gradient_accumulation_steps = 2
        self.dp = 0.1
        self.l2 = 1e-5
        self.l2_bert = 0.01
        self.warmup_proportion = 0.1
        self.adam_epsilon = 1e-8


