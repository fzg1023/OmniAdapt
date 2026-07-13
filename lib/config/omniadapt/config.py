from easydict import EasyDict as edict
import yaml

"""
Default config for OmniAdapt.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = ""
cfg.MODEL.EXTRA_MERGER = False
cfg.MODEL.RETURN_INTER = False
cfg.MODEL.RETURN_STAGES = []
cfg.MODEL.KEEP_RATE =  None
# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224"
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.MID_PE = False
cfg.MODEL.BACKBONE.SEP_SEG = False
cfg.MODEL.BACKBONE.CAT_MODE = 'direct'
cfg.MODEL.BACKBONE.MERGE_LAYER = 0
cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False
cfg.MODEL.BACKBONE.CLS_TOKEN_USE_MODE = 'ignore'

cfg.MODEL.BACKBONE.CROSS_LOC = []


# MODEL.CDA
cfg.MODEL.CDA = edict()
cfg.MODEL.CDA.OFFSET = 5.0
cfg.MODEL.CDA.TRACK_QUERY = 1
# MODEL.HEAD
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "CENTER"
cfg.MODEL.HEAD.NUM_CHANNELS = 256
cfg.MODEL.USE_RGF = True      # Sample-level RGF reliability/difference enhancements
cfg.MODEL.USE_AFG = True      # AdaptiveFusionGate
cfg.MODEL.USE_TOKEN_ADAPT = True  # Token-level SCA/aggregator/template-query/CSS

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.PARAM_KEY = False
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.FREEZE_LAYERS = [0, ]
cfg.TRAIN.FREEZE_EXCEPT = None   # Only train specified modules, freeze rest
cfg.TRAIN.FREEZE_LR = 0.0001
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False
## TRAIN save cfgs
cfg.TRAIN.FIX_BN = True
cfg.TRAIN.SAVE_EPOCH_INTERVAL = 1
cfg.TRAIN.SAVE_LAST_N_EPOCH = 1

cfg.TRAIN.DROP_PATH_RATE = 0.1
cfg.TRAIN.CROSS_DROP_PATH= [ 0.0, 0.0, 0.0 ]
# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

# DATA
cfg.DATA = edict()
cfg.DATA.SAMPLER_MODE = "causal"
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = []
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
cfg.DATA.SEARCH.NUMBER = 1
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.SEARCH_FACTOR = 4.0
cfg.TEST.SEARCH_SIZE = 256
cfg.TEST.EPOCH = 19
cfg.TEST.UPDATE_THRESHOLD = 0.65
cfg.TEST.UPDATE_INTERVALS = 5

# TEST.TCSR - Trajectory-Conditioned Score Refinement.
# Inference-time temporal prior; no extra training parameters are introduced.
cfg.TEST.TCSR = edict()
cfg.TEST.TCSR.ENABLE = False
cfg.TEST.TCSR.MODE = "linear"     # linear | kalman | advanced | rerank
cfg.TEST.TCSR.ALPHA = 0.10        # Base motion-prior strength.
cfg.TEST.TCSR.SIGMA = 2.0         # Base Gaussian width in score-map bins.
cfg.TEST.TCSR.BETA = 0.50         # Confidence-adaptive prior amplification.
cfg.TEST.TCSR.CONF_POWER = 1.0    # Shape of the confidence-to-alpha curve.
cfg.TEST.TCSR.MAX_ALPHA = 0.75    # Upper bound after adaptive amplification.
cfg.TEST.TCSR.UNCERTAINTY_SCALE = 1.0
cfg.TEST.TCSR.RERANK_TOPK = 5
cfg.TEST.TCSR.RERANK_LAMBDA = 0.20
cfg.TEST.TCSR.RERANK_CONF_MAX = 0.72
cfg.TEST.TCSR.RERANK_MARGIN = 0.04
cfg.TEST.TCSR.PROCESS_NOISE = 12.0
cfg.TEST.TCSR.MEASUREMENT_NOISE = 80.0

# TEST.TEMPORAL_GUARD - conservative gates for temporal priors and template updates.
cfg.TEST.TEMPORAL_GUARD = edict()
cfg.TEST.TEMPORAL_GUARD.ENABLE = True
cfg.TEST.TEMPORAL_GUARD.TCSR_CONF_MAX = 0.78
cfg.TEST.TEMPORAL_GUARD.TCSR_MARGIN_MAX = 0.05
cfg.TEST.TEMPORAL_GUARD.TCSR_MAX_INNOV = 2.50
cfg.TEST.TEMPORAL_GUARD.TEMPLATE_MARGIN_MIN = 0.03
cfg.TEST.TEMPORAL_GUARD.TEMPLATE_MAX_INNOV = 1.80
cfg.TEST.TEMPORAL_GUARD.TEMPLATE_MAX_SIZE_CHANGE = 0.50

# TEST.MEMORY - inference-time template memory.
cfg.TEST.MEMORY = edict()
cfg.TEST.MEMORY.ENABLE = False
cfg.TEST.MEMORY.BANK_SIZE = 5
cfg.TEST.MEMORY.SCORE_HIGH = 0.80
cfg.TEST.MEMORY.SCORE_LOW = 0.35
cfg.TEST.MEMORY.COOLDOWN = 10

# TEST.ROBUST_TEMPORAL - conservative inference-time recovery module.
# Disabled by default to keep the original std_full behavior reproducible.
cfg.TEST.ROBUST_TEMPORAL = edict()
cfg.TEST.ROBUST_TEMPORAL.ENABLE = False
cfg.TEST.ROBUST_TEMPORAL.DEBUG_LOG = False
cfg.TEST.ROBUST_TEMPORAL.DEBUG_LOG_PATH = ""
cfg.TEST.ROBUST_TEMPORAL.EARLY_FRAMES = 50
cfg.TEST.ROBUST_TEMPORAL.TOPK = 5
cfg.TEST.ROBUST_TEMPORAL.CONF_HIGH = 0.75
cfg.TEST.ROBUST_TEMPORAL.CONF_LOW = 0.35
cfg.TEST.ROBUST_TEMPORAL.MARGIN_HIGH = 0.08
cfg.TEST.ROBUST_TEMPORAL.MARGIN_LOW = 0.03
cfg.TEST.ROBUST_TEMPORAL.ENTROPY_HIGH = 1.01
cfg.TEST.ROBUST_TEMPORAL.RELIABILITY_HIGH = 0.70
cfg.TEST.ROBUST_TEMPORAL.RELIABILITY_LOW = 0.42
cfg.TEST.ROBUST_TEMPORAL.INNOV_GOOD = 0.80
cfg.TEST.ROBUST_TEMPORAL.INNOV_BAD = 2.20
cfg.TEST.ROBUST_TEMPORAL.SIZE_JUMP_MAX = 0.55
cfg.TEST.ROBUST_TEMPORAL.SMALL_AREA = 1024.0
cfg.TEST.ROBUST_TEMPORAL.SMALL_CONF_LOW = 0.45
cfg.TEST.ROBUST_TEMPORAL.UNCERTAIN_PATIENCE = 2
cfg.TEST.ROBUST_TEMPORAL.LOST_PATIENCE = 5
cfg.TEST.ROBUST_TEMPORAL.RECOVERY_PATIENCE = 2
cfg.TEST.ROBUST_TEMPORAL.STABLE_WINDOW = 5
cfg.TEST.ROBUST_TEMPORAL.VISUAL_WEIGHT = 1.00
cfg.TEST.ROBUST_TEMPORAL.MOTION_WEIGHT = 0.20
cfg.TEST.ROBUST_TEMPORAL.SCALE_WEIGHT = 0.10
cfg.TEST.ROBUST_TEMPORAL.STABILITY_WEIGHT = 0.15
cfg.TEST.ROBUST_TEMPORAL.SWITCH_MARGIN = 0.08
cfg.TEST.ROBUST_TEMPORAL.ACCEPT_MIN_GAIN = 0.12
cfg.TEST.ROBUST_TEMPORAL.FREEZE_UNCERTAIN = True
cfg.TEST.ROBUST_TEMPORAL.EXPANDED_SEARCH_ENABLE = False
cfg.TEST.ROBUST_TEMPORAL.EXPANDED_SEARCH_FACTOR = 6.0
cfg.TEST.ROBUST_TEMPORAL.EXPANDED_MIN_GAIN = 0.08
cfg.TEST.ROBUST_TEMPORAL.MEMORY_RECOVERY_ENABLE = False
cfg.TEST.ROBUST_TEMPORAL.MEMORY_BANK_SIZE = 5
cfg.TEST.ROBUST_TEMPORAL.MEMORY_CANDIDATE_SIZE = 3
cfg.TEST.ROBUST_TEMPORAL.MEMORY_SCORE_HIGH = 0.80
cfg.TEST.ROBUST_TEMPORAL.MEMORY_MARGIN_MIN = 0.05
cfg.TEST.ROBUST_TEMPORAL.MEMORY_COOLDOWN = 10


def _update_config(base_cfg, exp_cfg):
    if isinstance(exp_cfg, dict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if isinstance(v, dict) and isinstance(base_cfg[k], dict):
                    _update_config(base_cfg[k], v)
                else:
                    base_cfg[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))

def update_config_from_file(filename):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
    _update_config(cfg, exp_config)
