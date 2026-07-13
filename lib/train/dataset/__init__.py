try:
    from .lasot import Lasot
except ImportError:
    Lasot = None
try:
    from .got10k import Got10k
except ImportError:
    Got10k = None
try:
    from .tracking_net import TrackingNet
except ImportError:
    TrackingNet = None
try:
    from .imagenetvid import ImagenetVID
except ImportError:
    ImagenetVID = None
try:
    from .coco import MSCOCO
except ImportError:
    MSCOCO = None
try:
    from .coco_seq import MSCOCOSeq
except ImportError:
    MSCOCOSeq = None
try:
    from .got10k_lmdb import Got10k_lmdb
except ImportError:
    Got10k_lmdb = None
try:
    from .lasot_lmdb import Lasot_lmdb
except ImportError:
    Lasot_lmdb = None
try:
    from .imagenetvid_lmdb import ImagenetVID_lmdb
except ImportError:
    ImagenetVID_lmdb = None
try:
    from .coco_seq_lmdb import MSCOCOSeq_lmdb
except ImportError:
    MSCOCOSeq_lmdb = None
try:
    from .tracking_net_lmdb import TrackingNet_lmdb
except ImportError:
    TrackingNet_lmdb = None
# RGBT dataloader
from .lasher import LasHeR
from .vtuav import VTUAV
# RGBD dataloader
try:
    from .depthtrack import DepthTrack
except ImportError:
    DepthTrack = None
# Event dataloader
try:
    from .visevent import VisEvent
except ImportError:
    VisEvent = None
