#Config file
parameters = dict(
    name = "SGD",
    lr = 0.01 ,
    lr_scheduler = "constant",
    lr_decay = 0.8,
    stage_length = 100,
    staircase = False,
    clip_norm = 1,
    momentum = 0.9
    )


NUM_EPOCHS = 10
BATCH_SIZE = 8
TRAIN_SIZE = 32000
VAL_SIZE = 15000
MODE = 'train'

CHECKPOINTS_DIR = "/home/alemosan/lipReading/checkpoints/"
LOGS_DIR = "/home/alemosan/lipReading/logs/model_LRW_run_4/"
LOAD_MODEL = True
LOAD_PATH = "/home/alemosan/lipReading/checkpoints/"
LOAD_NAME = "model_LRW_run_4_11.tar"
SAVE_PATH = "/home/alemosan/lipReading/checkpoints/"
SAVE_NAME = "model_LRW_run_4"