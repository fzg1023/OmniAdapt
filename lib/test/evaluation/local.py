import os
from pathlib import Path

from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()
    project_root = Path(__file__).resolve().parents[3]
    prj_dir = os.environ.get('OMNIADAPT_ROOT', str(project_root))
    data_root = os.environ.get('DATA_ROOT', '')
    lasher_root = os.environ.get('LASHER_ROOT', os.path.join(data_root, 'lasher'))

    settings.davis_dir = ''
    settings.got10k_lmdb_path = os.path.join(data_root, 'got10k_lmdb')
    settings.got10k_path = os.path.join(data_root, 'got10k')
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = os.path.join(data_root, 'itb')
    settings.lasot_extension_subset_path_path = os.path.join(data_root, 'lasot_extension_subset')
    settings.lasot_lmdb_path = os.path.join(data_root, 'lasot_lmdb')
    settings.lasot_path = os.path.join(data_root, 'lasot')
    settings.network_path = os.path.join(prj_dir, 'output/test/networks')
    settings.nfs_path = os.path.join(data_root, 'nfs')
    settings.otb_path = os.path.join(data_root, 'otb')
    settings.prj_dir = prj_dir
    settings.result_plot_path = os.path.join(prj_dir, 'output/test/result_plots')
    settings.results_path = os.path.join(prj_dir, 'output/test/tracking_results')
    settings.save_dir = os.path.join(prj_dir, 'output')
    settings.segmentation_path = os.path.join(prj_dir, 'output/test/segmentation_results')
    settings.lasher_path = lasher_root
    settings.lashertestingset_path = os.path.join(lasher_root, 'testingset')
    settings.tc128_path = os.path.join(data_root, 'TC128')
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = os.path.join(data_root, 'tnl2k')
    settings.tpl_path = ''
    settings.trackingnet_path = os.path.join(data_root, 'trackingnet')
    settings.uav_path = os.path.join(data_root, 'uav')
    settings.vot18_path = os.path.join(data_root, 'vot2018')
    settings.vot22_path = os.path.join(data_root, 'vot2022')
    settings.vot_path = os.path.join(data_root, 'VOT2019')
    settings.youtubevos_dir = ''
    return settings
