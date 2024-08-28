"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.datasets.datasets.cfr_caption_datasets import (
    CFRDataset,
    CFREvalDataset,
)

from lavis.common.registry import registry



@registry.register_builder("CFR")
class CFRBuilder(BaseDatasetBuilder):
    train_dataset_cls = CFRDataset
    eval_dataset_cls = CFREvalDataset

    DATASET_CONFIG_DICT = {
        "flickr30k_entities": "configs/datasets/flickr30k_entities/defaults.yaml",
        "mscoco": "configs/datasets/mscoco/defaults.yaml",
    }


