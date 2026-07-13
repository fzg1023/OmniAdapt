import math
import numpy as np
from lib.models.omniadapt import build_omniadapt
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import PreprocessorMM
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class OmniAdapt(BaseTracker):
    _sequence_id_counter = 0
    def __init__(self, params):
        super(OmniAdapt, self).__init__(params)
        network = build_omniadapt(params.cfg, training=False)

        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
        print('Load pretrained model from: ' + self.params.checkpoint)
        identity_afg = (
            getattr(params, 'identity_afg', False)
            or os.environ.get('OMNIADAPT_IDENTITY_AFG', '0') == '1'
        )
        if identity_afg:
            n_afg = 0
            for module in network.modules():
                if module.__class__.__name__ == 'AdaptiveFusionGate':
                    module.GATE_SCALE = 0.0
                    n_afg += 1
            print(f'[Ablation] AFG forced to identity: {n_afg} module(s), GATE_SCALE=0')
        depth = 12
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = PreprocessorMM(mean= self.cfg.DATA.MEAN, std = self.cfg.DATA.STD)
        self.state = None
        self.num_template = self.cfg.DATA.TEMPLATE.NUMBER
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()
        self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS
        self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD

        # ── TCSR: 轨迹引导 Score 精修 ──────────────────────────────────
        _tcsr = getattr(self.cfg.TEST, 'TCSR', None)
        self.use_tcsr = getattr(_tcsr, 'ENABLE', False) if _tcsr is not None else False
        self.tcsr_mode = getattr(_tcsr, 'MODE', 'linear') if _tcsr is not None else 'linear'
        self.tcsr_mode = str(self.tcsr_mode).lower()
        self.tcsr_alpha = getattr(_tcsr, 'ALPHA', 0.10) if _tcsr is not None else 0.10
        self.tcsr_sigma = getattr(_tcsr, 'SIGMA', 2.0)  if _tcsr is not None else 2.0
        self.tcsr_beta = getattr(_tcsr, 'BETA', 0.50) if _tcsr is not None else 0.50
        self.tcsr_conf_power = getattr(_tcsr, 'CONF_POWER', 1.0) if _tcsr is not None else 1.0
        self.tcsr_max_alpha = getattr(_tcsr, 'MAX_ALPHA', 0.75) if _tcsr is not None else 0.75
        self.tcsr_uncertainty_scale = getattr(_tcsr, 'UNCERTAINTY_SCALE', 1.0) if _tcsr is not None else 1.0
        self.tcsr_rerank_topk = getattr(_tcsr, 'RERANK_TOPK', 5) if _tcsr is not None else 5
        self.tcsr_rerank_lambda = getattr(_tcsr, 'RERANK_LAMBDA', 0.20) if _tcsr is not None else 0.20
        self.tcsr_rerank_conf_max = getattr(_tcsr, 'RERANK_CONF_MAX', 0.72) if _tcsr is not None else 0.72
        self.tcsr_rerank_margin = getattr(_tcsr, 'RERANK_MARGIN', 0.04) if _tcsr is not None else 0.04
        self.tcsr_process_noise = getattr(_tcsr, 'PROCESS_NOISE', 12.0) if _tcsr is not None else 12.0
        self.tcsr_measurement_noise = getattr(_tcsr, 'MEASUREMENT_NOISE', 80.0) if _tcsr is not None else 80.0
        self.bbox_history = []
        self.temporal_state = None
        self.temporal_cov = None

        _guard = getattr(self.cfg.TEST, 'TEMPORAL_GUARD', None)
        self.temporal_guard_enable = getattr(_guard, 'ENABLE', True) if _guard is not None else True
        self.guard_tcsr_conf_max = getattr(_guard, 'TCSR_CONF_MAX', 0.78) if _guard is not None else 0.78
        self.guard_tcsr_margin_max = getattr(_guard, 'TCSR_MARGIN_MAX', 0.05) if _guard is not None else 0.05
        self.guard_tcsr_max_innov = getattr(_guard, 'TCSR_MAX_INNOV', 2.50) if _guard is not None else 2.50
        self.guard_template_margin_min = getattr(_guard, 'TEMPLATE_MARGIN_MIN', 0.03) if _guard is not None else 0.03
        self.guard_template_max_innov = getattr(_guard, 'TEMPLATE_MAX_INNOV', 1.80) if _guard is not None else 1.80
        self.guard_template_max_size_change = getattr(_guard, 'TEMPLATE_MAX_SIZE_CHANGE', 0.50) if _guard is not None else 0.50

        # ── 模板记忆库 (Plan C) ──────────────────────────────────────
        _mem = getattr(self.cfg.TEST, 'MEMORY', None)
        self.use_memory = getattr(_mem, 'ENABLE', False) if _mem is not None else False
        self.mem_bank_size  = getattr(_mem, 'BANK_SIZE', 5)  if _mem is not None else 5
        self.mem_score_high = getattr(_mem, 'SCORE_HIGH', 0.80) if _mem is not None else 0.80
        self.mem_score_low  = getattr(_mem, 'SCORE_LOW', 0.35)  if _mem is not None else 0.35
        self.mem_cooldown   = getattr(_mem, 'COOLDOWN', 10) if _mem is not None else 10
        self.memory_bank = []     # [(template_tensor, bbox_crop, score), ...]
        self.score_history = []   # 最近 20 帧 score
        self._mem_cooldown_counter = 0

        _rt = getattr(self.cfg.TEST, 'ROBUST_TEMPORAL', None)
        self.use_robust_temporal = getattr(_rt, 'ENABLE', False) if _rt is not None else False
        self.rt_debug_log = getattr(_rt, 'DEBUG_LOG', False) if _rt is not None else False
        self.rt_debug_log_path = getattr(_rt, 'DEBUG_LOG_PATH', '') if _rt is not None else ''
        self.rt_early_frames = getattr(_rt, 'EARLY_FRAMES', 50) if _rt is not None else 50
        self.rt_topk = getattr(_rt, 'TOPK', 5) if _rt is not None else 5
        self.rt_conf_high = getattr(_rt, 'CONF_HIGH', 0.75) if _rt is not None else 0.75
        self.rt_conf_low = getattr(_rt, 'CONF_LOW', 0.35) if _rt is not None else 0.35
        self.rt_margin_high = getattr(_rt, 'MARGIN_HIGH', 0.08) if _rt is not None else 0.08
        self.rt_margin_low = getattr(_rt, 'MARGIN_LOW', 0.03) if _rt is not None else 0.03
        self.rt_entropy_high = getattr(_rt, 'ENTROPY_HIGH', 1.01) if _rt is not None else 1.01
        self.rt_reliability_high = getattr(_rt, 'RELIABILITY_HIGH', 0.70) if _rt is not None else 0.70
        self.rt_reliability_low = getattr(_rt, 'RELIABILITY_LOW', 0.42) if _rt is not None else 0.42
        self.rt_innov_good = getattr(_rt, 'INNOV_GOOD', 0.80) if _rt is not None else 0.80
        self.rt_innov_bad = getattr(_rt, 'INNOV_BAD', 2.20) if _rt is not None else 2.20
        self.rt_size_jump_max = getattr(_rt, 'SIZE_JUMP_MAX', 0.55) if _rt is not None else 0.55
        self.rt_small_area = getattr(_rt, 'SMALL_AREA', 1024.0) if _rt is not None else 1024.0
        self.rt_small_conf_low = getattr(_rt, 'SMALL_CONF_LOW', 0.45) if _rt is not None else 0.45
        self.rt_uncertain_patience = getattr(_rt, 'UNCERTAIN_PATIENCE', 2) if _rt is not None else 2
        self.rt_lost_patience = getattr(_rt, 'LOST_PATIENCE', 5) if _rt is not None else 5
        self.rt_recovery_patience = getattr(_rt, 'RECOVERY_PATIENCE', 2) if _rt is not None else 2
        self.rt_stable_window = getattr(_rt, 'STABLE_WINDOW', 5) if _rt is not None else 5
        self.rt_visual_weight = getattr(_rt, 'VISUAL_WEIGHT', 1.0) if _rt is not None else 1.0
        self.rt_motion_weight = getattr(_rt, 'MOTION_WEIGHT', 0.20) if _rt is not None else 0.20
        self.rt_scale_weight = getattr(_rt, 'SCALE_WEIGHT', 0.10) if _rt is not None else 0.10
        self.rt_stability_weight = getattr(_rt, 'STABILITY_WEIGHT', 0.15) if _rt is not None else 0.15
        self.rt_switch_margin = getattr(_rt, 'SWITCH_MARGIN', 0.08) if _rt is not None else 0.08
        self.rt_accept_min_gain = getattr(_rt, 'ACCEPT_MIN_GAIN', 0.12) if _rt is not None else 0.12
        self.rt_freeze_uncertain = getattr(_rt, 'FREEZE_UNCERTAIN', True) if _rt is not None else True
        self.rt_expanded_search_enable = getattr(_rt, 'EXPANDED_SEARCH_ENABLE', False) if _rt is not None else False
        self.rt_expanded_search_factor = getattr(_rt, 'EXPANDED_SEARCH_FACTOR', 6.0) if _rt is not None else 6.0
        self.rt_expanded_min_gain = getattr(_rt, 'EXPANDED_MIN_GAIN', 0.08) if _rt is not None else 0.08
        self.rt_memory_recovery_enable = getattr(_rt, 'MEMORY_RECOVERY_ENABLE', False) if _rt is not None else False
        self.rt_memory_bank_size = getattr(_rt, 'MEMORY_BANK_SIZE', 5) if _rt is not None else 5
        self.rt_memory_candidate_size = getattr(_rt, 'MEMORY_CANDIDATE_SIZE', 3) if _rt is not None else 3
        self.rt_memory_score_high = getattr(_rt, 'MEMORY_SCORE_HIGH', 0.80) if _rt is not None else 0.80
        self.rt_memory_margin_min = getattr(_rt, 'MEMORY_MARGIN_MIN', 0.05) if _rt is not None else 0.05
        self.rt_memory_cooldown = getattr(_rt, 'MEMORY_COOLDOWN', 10) if _rt is not None else 10
        self.rt_state = 'TRACKING'
        self.rt_bad_count = 0
        self.rt_good_count = 0
        self.rt_low_conf_count = 0
        self.rt_recovery_count = 0
        self.rt_candidate_history = []
        self.rt_reliable_memory = []
        self.rt_candidate_memory = []
        self.rt_memory_cooldown_counter = 0
        self.rt_last_debug = None
        # ───────────────────────────────────────────────────────────────

        # for debug
        if getattr(params, 'debug', None) is None:
            setattr(params, 'debug', 0)
        self.use_visdom = False #params.debug
        self.debug = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                self._init_visdom(None, 1)

        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.sequence_id = OmniAdapt._sequence_id_counter
        OmniAdapt._sequence_id_counter += 1

    def initialize(self, image, info: dict):

        z_patch_arr, resize_factor, z_amask_arr  = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        
        template = self.preprocessor.process(z_patch_arr)
        with torch.no_grad():
            self.z_dict = [template]* self.num_template

        self.track_query_before = None

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        # ── TCSR: 初始化 bbox 历史（空列表，首帧由 track 填充）─────
        if self.use_tcsr or self.temporal_guard_enable or self.use_robust_temporal:
            self.bbox_history = []
            self._init_temporal_filter(info['init_bbox'])
        # ── 模板记忆库: 重置 ────────────────────────────────────────
        if self.use_memory:
            self.memory_bank = []
            self.score_history = []
            self._mem_cooldown_counter = 0
        if self.use_robust_temporal:
            self.rt_state = 'TRACKING'
            self.rt_bad_count = 0
            self.rt_good_count = 0
            self.rt_low_conf_count = 0
            self.rt_recovery_count = 0
            self.rt_candidate_history = [self._bbox_to_cxcywh(info['init_bbox']).tolist()]
            self.rt_reliable_memory = []
            self.rt_candidate_memory = []
            self.rt_memory_cooldown_counter = 0
            self.rt_last_debug = None
        # ───────────────────────────────────────────────────────────────
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr)

        # ── 模板记忆库: 低置信度时注入记忆模板 ──────────────────────
        templates_for_forward = list(self.z_dict)
        recent_avg = np.mean(self.score_history[-10:]) if len(self.score_history) >= 5 else 1.0
        mem_injected = False
        robust_mem_injected = False
        if self.use_robust_temporal:
            pass
        elif self.use_memory and len(self.memory_bank) > 0 and self._mem_cooldown_counter <= 0:
            if recent_avg < self.mem_score_low:
                best_mem = max(self.memory_bank, key=lambda x: x[2])[0]
                templates_for_forward.append(best_mem)
                mem_injected = True
        # ───────────────────────────────────────────────────────────────

        with torch.no_grad():
            x_dict = [search]
            if self.track_query_before != None:
                out_dict = self.network.forward(
                    template= templates_for_forward,
                    search=x_dict, track_query_before=self.track_query_before)
            else:
                out_dict = self.network.forward(
                    template= templates_for_forward,
                    search=x_dict)

        self.track_query_before  = out_dict[0]['track_query_before']

        pred_score_map = out_dict[0]['score_map']
        response = self.output_window * pred_score_map
        raw_conf_val = response.max().item()
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict[0]['size_map'], out_dict[0]['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) 裁剪像素坐标

        final_global_box = None
        robust_quality = None
        robust_selected = {'used_temporal': False, 'expanded': False, 'candidate_idx': 0}
        if self.use_robust_temporal:
            robust_quality = self._compute_response_quality(
                response, self.map_box_back(pred_box, resize_factor))
            pred_box, final_global_box, robust_quality, robust_selected = self._select_candidate_with_temporal_evidence(
                response, out_dict[0]['size_map'], out_dict[0]['offset_map'],
                resize_factor, raw_conf_val, pred_box, robust_quality)
            expanded = self._try_expanded_search_recovery(
                image, templates_for_forward, robust_quality, pred_box)
            if expanded is not None:
                exp_box, exp_global, exp_quality, exp_selected, exp_out = expanded
                pred_box = exp_box
                final_global_box = exp_global
                robust_quality = exp_quality
                robust_selected.update(exp_selected)
                if exp_out is not None:
                    out_dict = exp_out
                    self.track_query_before = out_dict[0]['track_query_before']
                    pred_score_map = out_dict[0]['score_map']
                    response = self.output_window * pred_score_map
                    raw_conf_val = response.max().item()
            memory_recovery = self._try_memory_recovery(image, robust_quality)
            if memory_recovery is not None:
                mem_box, mem_global, mem_quality, mem_selected, mem_out = memory_recovery
                pred_box = mem_box
                final_global_box = mem_global
                robust_quality = mem_quality
                robust_selected.update(mem_selected)
                robust_mem_injected = True
                if mem_out is not None:
                    out_dict = mem_out
                    self.track_query_before = out_dict[0]['track_query_before']
                    pred_score_map = out_dict[0]['score_map']
                    response = self.output_window * pred_score_map
                    raw_conf_val = response.max().item()

        # ── TCSR: 轨迹引导 Score 精修 ────────────────────────────────
        # 用旧历史 (不含当前帧) 推算运动先验, 避免循环
        temporal_candidate_box = self.map_box_back(pred_box, resize_factor)
        if (not self.use_robust_temporal and self.use_tcsr and self._can_apply_tcsr()
                and self._should_apply_temporal(response, raw_conf_val, temporal_candidate_box)):
            if self.tcsr_mode == 'rerank':
                pred_box = self._rerank_temporal_candidates(
                    response, out_dict[0]['size_map'], out_dict[0]['offset_map'],
                    resize_factor, raw_conf_val, pred_box)
            else:
                pred_score_map = self._apply_tcsr(pred_score_map, resize_factor, raw_conf_val)
                # 用精修后的 score 重新解码 bbox
                response2 = self.output_window * pred_score_map
                pred_boxes2 = self.network.box_head.cal_bbox(response2, out_dict[0]['size_map'], out_dict[0]['offset_map'])
                pred_boxes2 = pred_boxes2.view(-1, 4)
                pred_box = (pred_boxes2.mean(
                    dim=0) * self.params.search_size / resize_factor).tolist()
        # ───────────────────────────────────────────────────────────────

        # get the final box result
        if final_global_box is None:
            self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        else:
            self.state = clip_box(final_global_box, H, W, margin=10)
        if self.use_robust_temporal:
            if robust_quality is None:
                robust_quality = self._compute_response_quality(response, self.state)
            self._update_tracking_state(robust_quality)
            template_update_safe = self._should_update_template_safe(response, raw_conf_val, self.state, robust_quality)
        else:
            template_update_safe = self._is_template_update_safe(response, raw_conf_val, self.state)

        # ── TCSR: 更新全局坐标历史 (在精修之后, 为下一帧准备) ────────
        if self.use_robust_temporal:
            if self._robust_should_update_history(robust_quality):
                self._update_tcsr_state(self.state, raw_conf_val)
        elif self.use_tcsr or self.temporal_guard_enable:
            self._update_tcsr_state(self.state, raw_conf_val)
        # ───────────────────────────────────────────────────────────────

        # ── 模板记忆库: 更新 ────────────────────────────────────────
        if self.use_robust_temporal:
            self._safe_memory_update(image, self.state, robust_quality, robust_mem_injected)
        elif self.use_memory:
            conf_val = raw_conf_val
            self.score_history.append(conf_val)
            if len(self.score_history) > 30:
                self.score_history.pop(0)

            if mem_injected:
                self._mem_cooldown_counter = self.mem_cooldown
            elif self._mem_cooldown_counter > 0:
                self._mem_cooldown_counter -= 1

            # 高置信度 → 存入记忆
            if conf_val > self.mem_score_high and template_update_safe:
                mem_template = self._capture_template(image, self.state)
                self.memory_bank.append((mem_template, pred_box, conf_val))
                self.memory_bank.sort(key=lambda x: x[2], reverse=True)
                if len(self.memory_bank) > self.mem_bank_size:
                    self.memory_bank.pop()
        # ───────────────────────────────────────────────────────────────
        if self.use_robust_temporal:
            self._write_robust_debug(robust_quality, robust_selected, robust_mem_injected)
        conf_score = None
        if self.num_template > 1:
            conf_score, idx = torch.max(response.flatten(1), dim=1, keepdim=True)
            if ((self.frame_id % self.update_intervals == 0)
                    and (conf_score > self.update_threshold)
                    and template_update_safe):

                z_patch_arr, resize_factor, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                                        output_sz=self.params.template_size)
                self.z_patch_arr = z_patch_arr
                template = self.preprocessor.process(z_patch_arr)
                self.z_dict.append(template)
                if len(self.z_dict) > self.num_template:
                    self.z_dict.pop(1)

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap',
                                     1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            if conf_score != None:
                return {"target_bbox": self.state,
                        "all_boxes": all_boxes_save,
                        "best_score":conf_score.cpu().numpy()[0][0]}
            else:
                return {"target_bbox": self.state,
                        "all_boxes": all_boxes_save,
                        "best_score":None
                        }
        else:
            if conf_score != None:
                return {"target_bbox": self.state,
                        "best_score":conf_score.cpu().numpy()[0][0]
                        }
            else:
                return {"target_bbox": self.state,
                        "best_score":None
                        }

    def _clip01(self, value):
        return max(0.0, min(float(value), 1.0))

    def _bbox_size_jump(self, bbox):
        if not self.bbox_history:
            return 0.0
        prev = np.asarray(self.bbox_history[-1], dtype=np.float32)
        curr = self._bbox_to_cxcywh(bbox)
        prev_w, prev_h = max(float(prev[2]), 1.0), max(float(prev[3]), 1.0)
        curr_w, curr_h = max(float(curr[2]), 1.0), max(float(curr[3]), 1.0)
        return float(max(abs(math.log(curr_w / prev_w)), abs(math.log(curr_h / prev_h))))

    def _response_entropy(self, response):
        flat = response.flatten()
        if flat.numel() <= 1:
            return 0.0
        probs = torch.softmax(flat - flat.max(), dim=0)
        entropy = -(probs * torch.log(probs + 1e-12)).sum()
        return float(entropy / math.log(float(flat.numel())))

    def _compute_response_quality(self, response, candidate_box):
        flat = response.flatten(1)
        topk = torch.topk(flat, k=min(2, flat.shape[1]), dim=1).values[0]
        conf = float(topk[0])
        margin = float(topk[0] - topk[1]) if topk.numel() > 1 else 1.0
        entropy = self._response_entropy(response)
        innovation = self._normalized_innovation(candidate_box)
        size_jump = self._bbox_size_jump(candidate_box)
        area = max(float(candidate_box[2]), 0.0) * max(float(candidate_box[3]), 0.0)
        small_target = area <= float(self.rt_small_area)
        early = self.frame_id <= int(self.rt_early_frames)
        conf_score = self._clip01(conf / max(float(self.rt_conf_high), 1e-6))
        margin_score = self._clip01(margin / max(float(self.rt_margin_high), 1e-6))
        concentration = self._clip01(1.0 - entropy)
        innov_score = self._clip01(1.0 - innovation / max(float(self.rt_innov_bad), 1e-6))
        size_score = self._clip01(1.0 - size_jump / max(float(self.rt_size_jump_max), 1e-6))
        if small_target:
            reliability = (0.25 * conf_score + 0.30 * margin_score + 0.25 * concentration
                           + 0.10 * innov_score + 0.10 * size_score)
        else:
            reliability = (0.35 * conf_score + 0.20 * margin_score + 0.20 * concentration
                           + 0.15 * innov_score + 0.10 * size_score)
        low_conf = conf <= (float(self.rt_small_conf_low) if small_target else float(self.rt_conf_low))
        ambiguous = margin <= float(self.rt_margin_low) or entropy >= float(self.rt_entropy_high)
        bad_motion = innovation >= float(self.rt_innov_bad)
        bad_size = size_jump >= float(self.rt_size_jump_max)
        return {
            'confidence': conf,
            'margin': margin,
            'entropy': entropy,
            'innovation': innovation,
            'size_jump': size_jump,
            'area': area,
            'small_target': small_target,
            'early': early,
            'reliability': float(reliability),
            'low_conf': bool(low_conf),
            'ambiguous': bool(ambiguous),
            'bad_motion': bool(bad_motion),
            'bad_size': bool(bad_size),
        }

    def _decode_temporal_candidates(self, response, size_map, offset_map, resize_factor):
        boxes, scores = self._decode_topk_boxes(response, size_map, offset_map, self.rt_topk)
        boxes_px = boxes * (float(self.params.search_size) / max(float(resize_factor), 1e-6))
        candidate_global = [self.map_box_back(box.tolist(), resize_factor) for box in boxes_px]
        return boxes_px, scores.detach(), candidate_global

    def _temporal_history_reliable(self):
        return self.rt_state != 'LOST' and self.rt_bad_count < int(self.rt_lost_patience)

    def _candidate_scale_score(self, candidate):
        if not self.bbox_history:
            return 1.0
        prev = np.asarray(self.bbox_history[-1], dtype=np.float32)
        curr = self._bbox_to_cxcywh(candidate)
        pw, ph = max(float(prev[2]), 1.0), max(float(prev[3]), 1.0)
        cw, ch = max(float(curr[2]), 1.0), max(float(curr[3]), 1.0)
        jump = max(abs(math.log(cw / pw)), abs(math.log(ch / ph)))
        return self._clip01(1.0 - jump / max(float(self.rt_size_jump_max), 1e-6))

    def _candidate_stability_score(self, candidate):
        if not self.rt_candidate_history:
            return 0.5
        curr = self._bbox_to_cxcywh(candidate)
        prev = np.asarray(self.rt_candidate_history[-1], dtype=np.float32)
        scale = max(float(np.sqrt(prev[2] * prev[2] + prev[3] * prev[3])), 1.0)
        dist = float(np.linalg.norm(curr[:2] - prev[:2]) / scale)
        return float(math.exp(-0.5 * dist * dist))

    def _select_candidate_with_temporal_evidence(self, response, size_map, offset_map,
                                                 resize_factor, raw_conf, fallback_box,
                                                 fallback_quality):
        if (fallback_quality is not None
                and fallback_quality['reliability'] >= float(self.rt_reliability_high)
                and not fallback_quality['low_conf']
                and not fallback_quality['ambiguous']
                and not fallback_quality['bad_motion']
                and not fallback_quality['bad_size']):
            return fallback_box, None, fallback_quality, {
                'used_temporal': False, 'expanded': False, 'candidate_idx': 0}
        if fallback_quality is not None:
            need_intervention = (
                fallback_quality['bad_motion']
                or fallback_quality['bad_size']
                or self.rt_state == 'LOST'
            )
            if not need_intervention:
                return fallback_box, None, fallback_quality, {
                    'used_temporal': False, 'expanded': False, 'candidate_idx': 0}

        boxes_px, scores, candidate_global = self._decode_temporal_candidates(
            response, size_map, offset_map, resize_factor)
        if boxes_px.numel() == 0:
            return fallback_box, None, fallback_quality, {
                'used_temporal': False, 'expanded': False, 'candidate_idx': 0}

        score_vals = scores.detach()
        visual = score_vals / max(float(score_vals[0]), 1e-6)
        combined = visual * float(self.rt_visual_weight)
        used_temporal = False

        pred, pred_cov = self._predict_temporal_bbox() if self._can_apply_tcsr() else (None, None)
        if pred is not None and self._temporal_history_reliable():
            centers = torch.tensor(
                [[b[0] + 0.5 * b[2], b[1] + 0.5 * b[3]] for b in candidate_global],
                device=response.device, dtype=torch.float32)
            pred_center = torch.tensor([float(pred[0]), float(pred[1])],
                                       device=response.device, dtype=torch.float32)
            if pred_cov is not None:
                sigma_x = float(np.sqrt(max(pred_cov[0, 0], 1e-6)))
                sigma_y = float(np.sqrt(max(pred_cov[1, 1], 1e-6)))
            else:
                pred_w = max(float(pred[2] if len(pred) == 4 else pred[4]), 1.0)
                pred_h = max(float(pred[3] if len(pred) == 4 else pred[5]), 1.0)
                sigma_x = max(pred_w, 1.0)
                sigma_y = max(pred_h, 1.0)
            if fallback_quality is not None and fallback_quality.get('small_target', False):
                sigma_x *= 1.8
                sigma_y *= 1.8
            dx = (centers[:, 0] - pred_center[0]) / max(sigma_x, 1e-6)
            dy = (centers[:, 1] - pred_center[1]) / max(sigma_y, 1e-6)
            motion = torch.exp(-0.5 * (dx * dx + dy * dy))
            if fallback_quality is not None and (
                    fallback_quality['low_conf'] or fallback_quality['ambiguous']):
                combined = combined + float(self.rt_motion_weight) * motion
                used_temporal = True

        scale_scores = torch.tensor(
            [self._candidate_scale_score(b) for b in candidate_global],
            device=response.device, dtype=torch.float32)
        stability_scores = torch.tensor(
            [self._candidate_stability_score(b) for b in candidate_global],
            device=response.device, dtype=torch.float32)
        combined = combined + float(self.rt_scale_weight) * scale_scores
        combined = combined + float(self.rt_stability_weight) * stability_scores

        best = int(torch.argmax(combined).item())
        if fallback_quality is not None and fallback_quality['ambiguous'] and best != 0:
            gain = float(combined[best] - combined[0])
            if gain < float(self.rt_switch_margin):
                best = 0

        selected_global = candidate_global[best]
        selected_quality = self._compute_response_quality(response, selected_global)
        quality_gain = selected_quality['reliability'] - fallback_quality['reliability'] if fallback_quality is not None else 0.0
        if quality_gain < float(self.rt_accept_min_gain):
            return fallback_box, None, fallback_quality, {
                'used_temporal': used_temporal,
                'expanded': False,
                'candidate_idx': 0,
            }
        freeze = (
            self.rt_freeze_uncertain
            and selected_quality['reliability'] < float(self.rt_reliability_low)
            and (selected_quality['early'] or self.rt_state != 'TRACKING'
                 or selected_quality['bad_motion'])
        )
        if freeze:
            return fallback_box, list(self.state), selected_quality, {
                'used_temporal': used_temporal,
                'expanded': False,
                'candidate_idx': -1,
            }
        return boxes_px[best].tolist(), selected_global, selected_quality, {
            'used_temporal': used_temporal,
            'expanded': False,
            'candidate_idx': best,
        }

    def _update_tracking_state(self, quality):
        if quality is None:
            return
        if quality['confidence'] <= float(self.rt_conf_low):
            self.rt_low_conf_count += 1
        else:
            self.rt_low_conf_count = 0

        good = (
            quality['reliability'] >= float(self.rt_reliability_high)
            and not quality['bad_motion']
            and not quality['bad_size']
        )
        bad = (
            quality['reliability'] <= float(self.rt_reliability_low)
            or quality['bad_motion']
            or quality['bad_size']
            or self.rt_low_conf_count >= int(self.rt_uncertain_patience)
        )
        if good:
            self.rt_good_count += 1
            self.rt_bad_count = 0
            self.rt_recovery_count += 1
        elif bad:
            self.rt_bad_count += 1
            self.rt_good_count = 0
            self.rt_recovery_count = 0
        else:
            self.rt_good_count = 0
            self.rt_bad_count = max(0, self.rt_bad_count - 1)

        if self.rt_state == 'LOST':
            if self.rt_recovery_count >= int(self.rt_recovery_patience):
                self.rt_state = 'TRACKING'
        elif self.rt_bad_count >= int(self.rt_lost_patience):
            self.rt_state = 'LOST'
        elif (self.rt_bad_count >= int(self.rt_uncertain_patience)
              or (quality['ambiguous']
                  and quality['reliability'] < float(self.rt_reliability_high))):
            self.rt_state = 'UNCERTAIN'
        elif self.rt_good_count >= int(self.rt_recovery_patience):
            self.rt_state = 'TRACKING'

    def _should_update_template_safe(self, response, raw_conf, bbox, quality):
        if quality is None:
            return self._is_template_update_safe(response, raw_conf, bbox)
        if self.rt_state == 'LOST':
            return False
        if quality['early'] and quality['reliability'] < float(self.rt_reliability_high):
            return False
        if quality['reliability'] < float(self.rt_reliability_high):
            return False
        if quality['margin'] < float(self.rt_memory_margin_min):
            return False
        if quality['innovation'] > float(self.rt_innov_good):
            return False
        if quality['size_jump'] > float(self.rt_size_jump_max):
            return False
        return self._is_template_update_safe(response, raw_conf, bbox)

    def _robust_should_update_history(self, quality):
        if quality is None:
            return True
        if self.rt_state == 'LOST' and quality['reliability'] < float(self.rt_reliability_high):
            return False
        if quality['early'] and quality['reliability'] < float(self.rt_reliability_low):
            return False
        if quality['bad_motion'] or quality['bad_size']:
            return False
        self.rt_candidate_history.append(self._bbox_to_cxcywh(self.state).tolist())
        if len(self.rt_candidate_history) > int(self.rt_stable_window):
            self.rt_candidate_history.pop(0)
        return True

    def _robust_memory_templates(self):
        if self.rt_memory_cooldown_counter > 0:
            self.rt_memory_cooldown_counter -= 1
            return []
        if self.rt_state not in ('UNCERTAIN', 'LOST'):
            return []
        if not self.rt_reliable_memory:
            return []
        best = max(self.rt_reliable_memory, key=lambda x: x[2])
        self.rt_memory_cooldown_counter = int(self.rt_memory_cooldown)
        return [best[0]]

    def _try_memory_recovery(self, image, quality):
        if not self.rt_memory_recovery_enable:
            return None
        if quality is None or not self.rt_reliable_memory:
            return None
        if self.rt_state not in ('UNCERTAIN', 'LOST') and not quality['low_conf']:
            return None
        if self.rt_memory_cooldown_counter > 0:
            self.rt_memory_cooldown_counter -= 1
            return None
        best_mem = max(self.rt_reliable_memory, key=lambda x: x[2])[0]
        templates = list(self.z_dict) + [best_mem]
        x_patch_arr, resize_factor, _ = sample_target(
            image, self.state, self.params.search_factor, output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr)
        with torch.no_grad():
            x_dict = [search]
            if self.track_query_before is not None:
                out_dict = self.network.forward(
                    template=templates, search=x_dict,
                    track_query_before=self.track_query_before)
            else:
                out_dict = self.network.forward(template=templates, search=x_dict)
        score_map = out_dict[0]['score_map']
        response = self.output_window * score_map
        boxes = self.network.box_head.cal_bbox(
            response, out_dict[0]['size_map'], out_dict[0]['offset_map'])
        boxes = boxes.view(-1, 4)
        box = (boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
        global_box = self.map_box_back(box, resize_factor)
        mem_quality = self._compute_response_quality(response, global_box)
        gain = mem_quality['reliability'] - quality['reliability']
        if gain < float(self.rt_expanded_min_gain):
            return None
        self.rt_memory_cooldown_counter = int(self.rt_memory_cooldown)
        selected = {'used_temporal': False, 'expanded': False,
                    'memory_recovery': True, 'candidate_idx': 0}
        return box, global_box, mem_quality, selected, out_dict

    def _try_expanded_search_recovery(self, image, templates_for_forward, quality, fallback_box):
        if not self.rt_expanded_search_enable or quality is None:
            return None
        need_expand = self.rt_state == 'LOST'
        if not need_expand:
            return None
        factor = max(float(self.rt_expanded_search_factor), float(self.params.search_factor))
        if factor <= float(self.params.search_factor) + 1e-6:
            return None
        x_patch_arr, resize_factor, _ = sample_target(
            image, self.state, factor, output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr)
        with torch.no_grad():
            x_dict = [search]
            if self.track_query_before is not None:
                out_dict = self.network.forward(
                    template=templates_for_forward, search=x_dict,
                    track_query_before=self.track_query_before)
            else:
                out_dict = self.network.forward(template=templates_for_forward, search=x_dict)
        score_map = out_dict[0]['score_map']
        response = self.output_window * score_map
        boxes = self.network.box_head.cal_bbox(
            response, out_dict[0]['size_map'], out_dict[0]['offset_map'])
        boxes = boxes.view(-1, 4)
        box = (boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
        global_box = self.map_box_back(box, resize_factor)
        exp_quality = self._compute_response_quality(response, global_box)
        gain = exp_quality['reliability'] - quality['reliability']
        if gain < float(self.rt_expanded_min_gain):
            return None
        selected = {'used_temporal': False, 'expanded': True, 'candidate_idx': 0}
        return box, global_box, exp_quality, selected, out_dict

    def _safe_memory_update(self, image, bbox, quality, mem_injected):
        if mem_injected:
            self.rt_memory_cooldown_counter = int(self.rt_memory_cooldown)
        if quality is None:
            return
        if self.rt_state == 'LOST':
            return
        reliable = (
            quality['confidence'] >= float(self.rt_memory_score_high)
            and quality['margin'] >= float(self.rt_memory_margin_min)
            and quality['innovation'] <= float(self.rt_innov_good)
            and quality['size_jump'] <= float(self.rt_size_jump_max)
            and not (quality['early'] and quality['reliability'] < float(self.rt_reliability_high))
        )
        candidate = quality['reliability'] >= float(self.rt_reliability_high)
        if not (reliable or candidate):
            return
        mem_template = self._capture_template(image, bbox)
        item = (mem_template, list(bbox), float(quality['reliability']))
        if reliable:
            self.rt_reliable_memory.append(item)
            self.rt_reliable_memory.sort(key=lambda x: x[2], reverse=True)
            if len(self.rt_reliable_memory) > int(self.rt_memory_bank_size):
                self.rt_reliable_memory.pop()
        else:
            self.rt_candidate_memory.append(item)
            self.rt_candidate_memory.sort(key=lambda x: x[2], reverse=True)
            if len(self.rt_candidate_memory) > int(self.rt_memory_candidate_size):
                self.rt_candidate_memory.pop()

    def _write_robust_debug(self, quality, selected, mem_injected):
        if not self.rt_debug_log or quality is None:
            return
        log_path = self.rt_debug_log_path
        if not log_path:
            log_path = os.path.join('debug', f'robust_temporal_{self.sequence_id}.csv')
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        need_header = not os.path.exists(log_path)
        with open(log_path, 'a') as f:
            if need_header:
                f.write('frame,state,confidence,margin,entropy,innovation,size_jump,'
                        'reliability,used_temporal,expanded,memory_recovery,candidate_idx\n')
            f.write('{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{},{},{},{}\n'.format(
                self.frame_id, self.rt_state, quality['confidence'], quality['margin'],
                quality['entropy'], quality['innovation'], quality['size_jump'],
                quality['reliability'], int(selected.get('used_temporal', False)),
                int(selected.get('expanded', False)), int(mem_injected),
                selected.get('candidate_idx', 0)))

    def _capture_template(self, image, bbox):
        """捕获当前帧目标区域作为模板 tensor"""
        z_patch, _, _ = sample_target(image, bbox, self.params.template_factor,
                                       output_sz=self.params.template_size)
        return self.preprocessor.process(z_patch)

    def _bbox_to_cxcywh(self, bbox):
        x, y, w, h = [float(v) for v in bbox[:4]]
        return np.array([x + 0.5 * w, y + 0.5 * h, w, h], dtype=np.float32)

    def _init_temporal_filter(self, bbox):
        cx, cy, w, h = self._bbox_to_cxcywh(bbox)
        self.bbox_history = [[cx, cy, w, h]]
        self.temporal_state = np.array([cx, cy, 0.0, 0.0, w, h], dtype=np.float32)
        self.temporal_cov = np.diag([100.0, 100.0, 400.0, 400.0, 100.0, 100.0]).astype(np.float32)

    def _is_filter_tcsr(self):
        return self.tcsr_mode in ('kalman', 'ktpr', 'advanced', 'rerank')

    def _can_apply_tcsr(self):
        if self._is_filter_tcsr():
            return self.temporal_state is not None
        return len(self.bbox_history) >= 2

    def _predict_temporal_bbox(self):
        if self._is_filter_tcsr():
            if self.temporal_state is None:
                return None, None
            F = np.eye(6, dtype=np.float32)
            F[0, 2] = 1.0
            F[1, 3] = 1.0
            q = float(max(self.tcsr_process_noise, 1e-3))
            Q = np.diag([0.25 * q, 0.25 * q, q, q, 0.10 * q, 0.10 * q]).astype(np.float32)
            pred_state = F @ self.temporal_state
            pred_cov = F @ self.temporal_cov @ F.T + Q
            return pred_state, pred_cov

        prev = np.array(self.bbox_history[-2], dtype=np.float32)
        curr = np.array(self.bbox_history[-1], dtype=np.float32)
        velocity = curr[:2] - prev[:2]
        pred = curr.copy()
        pred[:2] = curr[:2] + velocity
        return pred, None

    def _global_to_score_point(self, global_cx, global_cy, resize_factor, score_h, score_w):
        crop_cx = float(self.state[0] + 0.5 * self.state[2])
        crop_cy = float(self.state[1] + 0.5 * self.state[3])
        crop_side = float(self.params.search_size) / max(float(resize_factor), 1e-6)
        left = crop_cx - 0.5 * crop_side
        top = crop_cy - 0.5 * crop_side
        score_x = ((float(global_cx) - left) / crop_side) * score_w
        score_y = ((float(global_cy) - top) / crop_side) * score_h
        return score_x, score_y, crop_side

    def _score_peak_margin(self, response):
        flat = response.flatten(1)
        if flat.shape[1] < 2:
            return 1.0
        top2 = torch.topk(flat, k=2, dim=1).values[0]
        return float(top2[0] - top2[1])

    def _normalized_innovation(self, bbox, pred=None):
        if pred is None:
            if not self._can_apply_tcsr():
                return 0.0
            pred, _ = self._predict_temporal_bbox()
        if pred is None:
            return 0.0
        meas = self._bbox_to_cxcywh(bbox)
        pred_w = max(float(pred[4] if len(pred) > 4 else pred[2]), 1.0)
        pred_h = max(float(pred[5] if len(pred) > 5 else pred[3]), 1.0)
        scale = max(float(np.sqrt(pred_w * pred_w + pred_h * pred_h)), 1.0)
        return float(np.linalg.norm(meas[:2] - np.asarray(pred[:2], dtype=np.float32)) / scale)

    def _should_apply_temporal(self, response, raw_conf, candidate_box):
        if not self.temporal_guard_enable:
            return True
        margin = self._score_peak_margin(response)
        low_conf = float(raw_conf) <= float(self.guard_tcsr_conf_max)
        ambiguous = margin <= float(self.guard_tcsr_margin_max)
        if not (low_conf or ambiguous):
            return False
        if self._normalized_innovation(candidate_box) > float(self.guard_tcsr_max_innov):
            return False
        return True

    def _is_template_update_safe(self, response, raw_conf, bbox):
        if not self.temporal_guard_enable:
            return True
        if self._score_peak_margin(response) < float(self.guard_template_margin_min):
            return False
        if self._normalized_innovation(bbox) > float(self.guard_template_max_innov):
            return False
        if self.bbox_history:
            prev = np.asarray(self.bbox_history[-1], dtype=np.float32)
            curr = self._bbox_to_cxcywh(bbox)
            prev_w, prev_h = max(float(prev[2]), 1.0), max(float(prev[3]), 1.0)
            curr_w, curr_h = max(float(curr[2]), 1.0), max(float(curr[3]), 1.0)
            size_jump = max(abs(math.log(curr_w / prev_w)), abs(math.log(curr_h / prev_h)))
            if size_jump > float(self.guard_template_max_size_change):
                return False
        return True

    def _decode_topk_boxes(self, response, size_map, offset_map, topk):
        _, _, H, W = response.shape
        flat = response.flatten(1)
        k = min(max(int(topk), 1), flat.shape[1])
        scores, idx = torch.topk(flat, k=k, dim=1)
        idx_y = idx // W
        idx_x = idx % W
        gather_idx = idx.unsqueeze(1).expand(idx.shape[0], 2, k)
        size = size_map.flatten(2).gather(dim=2, index=gather_idx).transpose(1, 2)
        offset = offset_map.flatten(2).gather(dim=2, index=gather_idx).transpose(1, 2)
        boxes = torch.cat([
            (idx_x.to(torch.float32).unsqueeze(-1) + offset[:, :, :1]) / float(W),
            (idx_y.to(torch.float32).unsqueeze(-1) + offset[:, :, 1:]) / float(H),
            size
        ], dim=-1)
        return boxes[0], scores[0]

    def _rerank_temporal_candidates(self, response, size_map, offset_map, resize_factor, raw_conf, fallback_box):
        pred, pred_cov = self._predict_temporal_bbox()
        if pred is None:
            return fallback_box

        boxes, scores = self._decode_topk_boxes(response, size_map, offset_map, self.tcsr_rerank_topk)
        if boxes.numel() == 0:
            return fallback_box

        score_vals = scores.detach()
        margin = float(score_vals[0] - score_vals[1]) if score_vals.numel() > 1 else 1.0
        low_conf = float(raw_conf) <= float(self.tcsr_rerank_conf_max)
        ambiguous = margin <= float(self.tcsr_rerank_margin)
        if not (low_conf or ambiguous):
            return fallback_box

        boxes_px = boxes * (float(self.params.search_size) / max(float(resize_factor), 1e-6))
        candidate_global = [self.map_box_back(box.tolist(), resize_factor) for box in boxes_px]
        centers = torch.tensor(
            [[b[0] + 0.5 * b[2], b[1] + 0.5 * b[3]] for b in candidate_global],
            device=response.device, dtype=torch.float32)
        pred_center = torch.tensor([float(pred[0]), float(pred[1])], device=response.device, dtype=torch.float32)

        if pred_cov is not None:
            sigma_x = float(np.sqrt(max(pred_cov[0, 0], 1e-6)))
            sigma_y = float(np.sqrt(max(pred_cov[1, 1], 1e-6)))
            pred_w = max(float(pred[4]), 1.0)
            pred_h = max(float(pred[5]), 1.0)
        else:
            sigma_x = sigma_y = float(max(self.tcsr_sigma, 1.0) * self.params.search_size / self.feat_sz)
            pred_w = pred_h = float(self.params.search_size) / max(float(resize_factor), 1e-6)
        sigma_x = max(sigma_x, pred_w * 0.30)
        sigma_y = max(sigma_y, pred_h * 0.30)

        dx = (centers[:, 0] - pred_center[0]) / max(sigma_x, 1e-6)
        dy = (centers[:, 1] - pred_center[1]) / max(sigma_y, 1e-6)
        motion_score = torch.exp(-0.5 * (dx * dx + dy * dy))
        visual_score = score_vals / max(float(score_vals[0]), 1e-6)
        combined = visual_score + float(self.tcsr_rerank_lambda) * motion_score
        best = int(torch.argmax(combined).item())
        return boxes_px[best].tolist()

    def _apply_tcsr(self, score_map, resize_factor, raw_conf):
        """Temporal score refinement with either linear or Kalman motion prior."""
        pred, pred_cov = self._predict_temporal_bbox()
        if pred is None:
            return score_map

        _, _, H, W = score_map.shape
        pred_x, pred_y, crop_side = self._global_to_score_point(pred[0], pred[1], resize_factor, H, W)
        if pred_x < -W or pred_x > 2 * W or pred_y < -H or pred_y > 2 * H:
            return score_map

        sigma = float(max(self.tcsr_sigma, 0.5))
        sigma_x_base = sigma
        sigma_y_base = sigma
        if pred_cov is not None:
            unc_scale = max(float(self.tcsr_uncertainty_scale), 0.0)
            if self.tcsr_mode == 'advanced':
                unc_x = float(np.sqrt(max(pred_cov[0, 0], 1e-6)))
                unc_y = float(np.sqrt(max(pred_cov[1, 1], 1e-6)))
                sigma_x_base += unc_scale * (unc_x / max(crop_side, 1e-6)) * float(W)
                sigma_y_base += unc_scale * (unc_y / max(crop_side, 1e-6)) * float(H)
            else:
                pos_unc = float(np.sqrt(max(pred_cov[0, 0] + pred_cov[1, 1], 1e-6)) * 0.5)
                sigma_x_base += unc_scale * (pos_unc / max(crop_side, 1e-6)) * float(W)
                sigma_y_base += unc_scale * (pos_unc / max(crop_side, 1e-6)) * float(H)
        if self._is_filter_tcsr():
            pred_w = max(float(pred[4]), 1.0)
            pred_h = max(float(pred[5]), 1.0)
        else:
            pred_w = max(float(pred[2]), 1.0)
            pred_h = max(float(pred[3]), 1.0)
        sigma_x = max(sigma_x_base, 0.25 * pred_w / max(crop_side, 1e-6) * W)
        sigma_y = max(sigma_y_base, 0.25 * pred_h / max(crop_side, 1e-6) * H)
        sigma_x = min(max(sigma_x, 0.5), float(W))
        sigma_y = min(max(sigma_y, 0.5), float(H))

        ys = torch.arange(H, device=score_map.device, dtype=torch.float32).view(-1, 1)
        xs = torch.arange(W, device=score_map.device, dtype=torch.float32).view(1, -1)
        dist = ((xs - pred_x) ** 2) / (2.0 * sigma_x * sigma_x) + ((ys - pred_y) ** 2) / (2.0 * sigma_y * sigma_y)
        prior = torch.exp(-dist).unsqueeze(0).unsqueeze(0)

        alpha = max(0.0, min(float(self.tcsr_alpha), 0.5))
        if self._is_filter_tcsr():
            conf = max(0.0, min(float(raw_conf), 1.0))
            conf_gap = (1.0 - conf) ** max(float(self.tcsr_conf_power), 1e-3)
            alpha = alpha * (1.0 + max(0.0, float(self.tcsr_beta)) * conf_gap)
        alpha = max(0.0, min(alpha, max(float(self.tcsr_max_alpha), 0.0)))
        return score_map * (1.0 + alpha * prior)

    def _update_tcsr_state(self, bbox, raw_conf):
        meas = self._bbox_to_cxcywh(bbox)
        self.bbox_history.append(meas.tolist())
        if len(self.bbox_history) > 5:
            self.bbox_history.pop(0)

        if not self._is_filter_tcsr():
            return
        if self.temporal_state is None or self.temporal_cov is None:
            self._init_temporal_filter(bbox)
            return

        pred_state, pred_cov = self._predict_temporal_bbox()
        Hm = np.zeros((4, 6), dtype=np.float32)
        Hm[0, 0] = 1.0
        Hm[1, 1] = 1.0
        Hm[2, 4] = 1.0
        Hm[3, 5] = 1.0

        conf = max(0.0, min(float(raw_conf), 1.0))
        meas_noise = float(max(self.tcsr_measurement_noise, 1e-3)) * (1.5 - 0.5 * conf)
        R = np.diag([meas_noise, meas_noise, 0.5 * meas_noise, 0.5 * meas_noise]).astype(np.float32)
        innovation = meas - Hm @ pred_state
        S = Hm @ pred_cov @ Hm.T + R
        K = pred_cov @ Hm.T @ np.linalg.inv(S)
        self.temporal_state = (pred_state + K @ innovation).astype(np.float32)
        self.temporal_cov = ((np.eye(6, dtype=np.float32) - K @ Hm) @ pred_cov).astype(np.float32)

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return OmniAdapt
