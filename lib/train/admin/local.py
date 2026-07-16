import os


class EnvironmentSettings:
    def __init__(self):
        workspace_dir = os.environ.get('OMNIADAPT_ROOT', '/home/sau_fzg1/OmniAdapt')
        data_root = os.environ.get('DATA_ROOT', '/home/sau_fzg1/data')
        lasher_root = os.environ.get('LASHER_ROOT', os.path.join(data_root, 'lasher'))

        self.workspace_dir = workspace_dir
        self.tensorboard_dir = os.path.join(workspace_dir, 'tensorboard')
        self.pretrained_networks = os.path.join(workspace_dir, 'pretrain')
        self.got10k_val_dir = os.path.join(data_root, 'got10k/val')
        self.lasot_lmdb_dir = os.path.join(data_root, 'lasot_lmdb')
        self.got10k_lmdb_dir = os.path.join(data_root, 'got10k_lmdb')
        self.trackingnet_lmdb_dir = os.path.join(data_root, 'trackingnet_lmdb')
        self.coco_lmdb_dir = os.path.join(data_root, 'coco_lmdb')
        self.coco_dir = os.path.join(data_root, 'coco')
        self.lasot_dir = os.path.join(data_root, 'lasot')
        self.got10k_dir = os.path.join(data_root, 'got10k/train')
        self.trackingnet_dir = os.path.join(data_root, 'trackingnet')
        self.depthtrack_dir = os.path.join(data_root, 'depthtrack/train')
        self.lasher_dir = os.path.join(lasher_root, 'trainingset')
        self.vtuav_dir = os.path.join(data_root, 'VTUAV')
        self.visevent_dir = os.path.join(data_root, 'visevent/train')
