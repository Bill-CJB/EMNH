USE_CUDA = True
CUDA_DEVICE_NUM = 7
SEED = 1234

PROBLEM_SIZE = 20  # number of customer nodes

MODE = 2  # 1 denotes Train, 2 denotes Finetune and Test, 3 denotes Test all finetune from meta
METHOD = 'EMNH'
SAVE_NUM = 150

# LOAD_PATH = 'result/size20/checkpoint-1.pt'
LOAD_PATH = None
MODEL_DIR = 'result/size20'

# Hyper-Parameters
TOTAL_EPOCH = 3000
UPDATE_STEP = 100
FINETUNE_STEP = 20
TASK_NUM = 2
N_WEIGHT = 101
AGG = 1  # Weight Aggregation, 1 denotes Weighted-Sum, 2 denotes Weighted-Tchebycheff
TEST_DATASET_SIZE = 200
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 200
TESTAUG_BATCH_SIZE = 200

EMBEDDING_DIM = 128
KEY_DIM = 16  # Length of q, k, v of EACH attention head
HEAD_NUM = 8
ENCODER_LAYER_NUM = 6
FF_HIDDEN_DIM = 512
LOGIT_CLIPPING = 10  # (C in the paper)

META_LR = 1
ACTOR_LEARNING_RATE = 1e-4
ACTOR_WEIGHT_DECAY = 1e-6

LR_DECAY_EPOCH = 1
LR_DECAY_GAMMA = 1.00

# Logging
LOG_PERIOD_SEC = 15
