"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
import torch

from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask


@registry.register_task("captioning")
class CaptionTask(BaseTask):
    def __init__(self, cfg, if_sample, num_beams, max_len, min_len, evaluate, report_metric=True):
        super().__init__()

        self.cfg = cfg

        self.if_sample = if_sample
        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate

        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        if_sample = run_cfg.if_sample
        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            cfg=cfg,
            if_sample=if_sample,
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
        )

    def valid_step(self, model, samples):
        '''
            samples: dict_keys(['image_list', 'id_list', 'ann_list'])
        '''
        results = []

        captions = model.generate(
            samples,
            use_nucleus_sampling=self.if_sample,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
            num_captions=self.cfg.run_cfg['gen_num'],
        )

        if self.cfg.run_cfg['gen_num'] > 1:
            captions = [captions[i:i+self.cfg.run_cfg['gen_num']] for i in range(0, len(captions), self.cfg.run_cfg['gen_num'])]

        img_ids = samples["id_list"].cpu().numpy().tolist()
        if self.cfg.run_cfg.get("generate_on_train", False) or self.cfg.run_cfg.get("if_mask", False):
            for caption, img_id, ann in zip(captions, img_ids, samples["ann_list"]):
                results.append({"caption": caption, "image_id": int(img_id), "ann": ann})
        else:
            for caption, img_id in zip(captions, img_ids):
                results.append({"caption": caption, "image_id": int(img_id)})

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        if self.cfg.run_cfg.get("if_mask", False):
            filename = "mask_res_gen_num5"
        filename = "{}_epoch{}".format(split_name, epoch)
        
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename=filename,
            remove_duplicate="image_id" if not self.cfg.run_cfg.get("evaluate", False) else "",
        )

        if self.report_metric and not self.cfg.run_cfg.get("generate_on_train", False) \
            and not self.cfg.run_cfg.get("if_mask", False):
            metrics = self._report_metrics(
                eval_result_file=eval_result_file, split_name=split_name
            )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        # result_dir = self.cfg.run_cfg['result_dir']
        # if not os.path.exists(result_dir):
        #     os.makedirs(result_dir)

        # TODO better way to define this
        # coco_gt_root = os.path.join(registry.get_path("cache_root"), "coco_gt")
        
        coco_val = coco_caption_eval(
            self.cfg.run_cfg['caption_gt_root'],
            self.cfg.datasets_cfg[list(self.cfg.datasets_cfg.keys())[0]]['type'],
            eval_result_file,
            split_name
        )

        agg_metrics = coco_val.eval["CIDEr"] + coco_val.eval["Bleu_4"]
        log_stats = {split_name: {k: v for k, v in coco_val.eval.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in coco_val.eval.items()}
        coco_res["agg_metrics"] = agg_metrics

        return coco_res


# TODO better structure for this.
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from torchvision.datasets.utils import download_url


def coco_caption_eval(caption_gt_root, dataset_name, results_file, split):
    annotation_file = os.path.join(caption_gt_root, f'{dataset_name}_{split}_gt.json')
    
    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    print(coco_eval.eval.items())
    
    return coco_eval







