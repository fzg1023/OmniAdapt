import os

from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    prj_dir = os.environ.get('OMNIADAPT_ROOT', '/home/sau_fzg1/OmniAdapt')
    data_root = os.environ.get('DATA_ROOT', '/home/sau_fzg1/data')
    lasher_root = os.environ.get('LASHER_ROOT', os.path.join(data_root, 'lasher'))

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/fzg/CAD/data/got10k_lmdb'
    settings.got10k_path = '/home/fzg/CAD/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/fzg/CAD/data/itb'
    settings.lasot_extension_subset_path_path = '/home/fzg/CAD/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/fzg/CAD/data/lasot_lmdb'
    settings.lasot_path = '/home/fzg/CAD/data/lasot'
    settings.network_path = os.path.join(prj_dir, 'output/test/networks')    # Where tracking networks are stored.
    settings.nfs_path = '/home/fzg/CAD/data/nfs'
    settings.otb_path = '/home/fzg/CAD/data/otb'
    settings.prj_dir = prj_dir
    settings.result_plot_path = os.path.join(prj_dir, 'output/test/result_plots')
    settings.results_path = os.path.join(prj_dir, 'output/test/tracking_results')    # Where to store tracking results
    settings.save_dir = os.path.join(prj_dir, 'output')
    settings.segmentation_path = os.path.join(prj_dir, 'output/test/segmentation_results')
    settings.lasher_path = lasher_root
    settings.lashertestingset_path = os.path.join(lasher_root, 'testingset')
    settings.tc128_path = '/home/fzg/CAD/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/fzg/CAD/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/fzg/CAD/data/trackingnet'
    settings.uav_path = '/home/fzg/CAD/data/uav'
    settings.vot18_path = '/home/fzg/CAD/data/vot2018'
    settings.vot22_path = '/home/fzg/CAD/data/vot2022'
    settings.vot_path = '/home/fzg/CAD/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings
