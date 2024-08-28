"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import pickle
import random
from PIL import Image
from PIL import ImageFile, ImageDraw, ImageFilter

ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate

# from lavis.datasets.datasets.caption_datasets import CaptionDataset, CaptionEvalDataset
from lavis.datasets.datasets.dataloader_utils import cap_train_collate_fn, cap_eval_collate_fn


def blacken_box(image, boxes):
    '''
    This function blackens the boxes in the image and applies Gaussian blur to smooth the edges.
    '''
    # Create an ImageDraw object for modifying the image
    draw = ImageDraw.Draw(image)
    for box in boxes:
        # Get the coordinates of the rectangle box
        x1, y1, x2, y2 = box
        
        # Expand the boundary of the rectangle box
        expand_by = 3
        x1_expanded = max(0, x1 - expand_by)
        y1_expanded = max(0, y1 - expand_by)
        x2_expanded = min(image.width, x2 + expand_by)
        y2_expanded = min(image.height, y2 + expand_by)
        # box_expanded = (x1_expanded, y1_expanded, x2_expanded, y2_expanded)
        ### change into integer
        box_expanded = (int(x1_expanded), int(y1_expanded), int(x2_expanded), int(y2_expanded))

        # Fill in black pixels in the rectangle box
        draw.rectangle((x1, y1, x2, y2), fill='black')
        
        # Apply Gaussian blur to smooth the edges
        blur_radius = 5
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Paste the blurred black rectangle box on the original image
        image.paste(blurred_image.crop(box_expanded), box_expanded)
        
    return image


class CFRDataset(Dataset):
    def __init__(self, all_config, vis_processor, text_processor, vis_root, split, mask=False):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        split (string): train, val or test
        """
        super().__init__()
        
        self.dataset_name = all_config.datasets_cfg[list(all_config.datasets_cfg.keys())[0]]['type']
        self.all_config = all_config
        self.collater = cap_train_collate_fn if split == 'train' and not all_config.run_cfg['generate_on_train'] else cap_eval_collate_fn

        self.image_size = all_config.datasets_cfg['CFR']['vis_processor'][split]['image_size']
        self.max_words = all_config.run_cfg['max_len']
        self.prompt = all_config.datasets_cfg['CFR']['text_processor'][split]['prompt']

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        ##### for total effect
        self.do_Total_Effect = all_config.run_cfg['do_TE']
        self.do_NDE = all_config.run_cfg['do_NDE']

        self.base_path = vis_root

        split_file = self.all_config.datasets_cfg['CFR']['build_info']['annotations'][split]['url']
        self.annotation = pickle.load(open(split_file, 'rb'))
        
        if self.do_Total_Effect or self.do_NDE:
            gen_res_on_train_path = os.path.join(all_config.run_cfg['caption_gt_root'], 'gen_res_train/') + 'train_epochbest.json'
            self.gen_res_on_train = json.load(open(gen_res_on_train_path, 'r'))
            self.all_img_id2gen_res = {}
            for res in self.gen_res_on_train:
                if res['image_id'] not in self.all_img_id2gen_res: # res['image_id'] is an int
                    self.all_img_id2gen_res[res['image_id']] = [res]
                else:
                    self.all_img_id2gen_res[res['image_id']].append(res)
            print(f'In [Causal_Effect] training mode: TE={self.do_Total_Effect}, NDE={self.do_NDE}')
            print(f'Number of all_img_id2gen_res: {len(self.all_img_id2gen_res)}')

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]

        if self.dataset_name == 'flickr30k_entities':
            image_path = os.path.join(self.base_path + 'flickr30k-images/', ann['img_id']+'.jpg')
        elif self.dataset_name == 'mscoco':
            image_path = os.path.join(self.base_path, ann['img_path'])
        
        image_ori = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image_ori)
        
        caption = self.text_processor(ann['caption'])

        masked_images, gen_caps, entities, prefixes = [], [], [], []
        if self.do_Total_Effect or self.do_NDE:
            # get the generated results of the masked image
            if int(ann['img_id']) in self.all_img_id2gen_res:
                gen_res = self.all_img_id2gen_res[int(ann['img_id'])] # This might be multiple
                for res in gen_res:
                    gen_cap = self.text_processor(res['caption'])
                    gen_caps.append(gen_cap)
                    entity = self.text_processor(res['ann']['phrases']['phrase'])
                    entities.append(entity)
                    pre_gt_cap = self.text_processor(res['ann']['caption'])
                    prefix_list = pre_gt_cap.split(' ')[:res['ann']['phrases']['first_word_index']] if res['ann']['phrases']['first_word_index'] != 0 else []
                    prefix = ' '.join(prefix_list)
                    prefixes.append(prefix)
                    masked_image = blacken_box(image_ori.copy(), res['ann']['phrases']['boxes'])
                    masked_images.append(self.vis_processor(masked_image))
            
        return image, caption, masked_images, gen_caps, entities, prefixes



class CFREvalDataset(Dataset):
    def __init__(self, all_config, vis_processor, text_processor, vis_root, split, mask=False):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        split (string): val or test
        """
        super().__init__()

        self.dataset_name = all_config.datasets_cfg[list(all_config.datasets_cfg.keys())[0]]['type']
        self.all_config = all_config
        self.collater = cap_eval_collate_fn

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.generate_on_train = all_config.run_cfg['generate_on_train']
        if self.generate_on_train:
            self.image_size = all_config.datasets_cfg['CFR']['vis_processor']['test']['image_size']
        else:
            self.image_size = all_config.datasets_cfg['CFR']['vis_processor'][split]['image_size']
        
        if mask:
            self.generate_on_train = True
        
        self.base_path = vis_root

        split_file = self.all_config.datasets_cfg['CFR']['build_info']['annotations'][split]['url']
        self.annotation = pickle.load(open(split_file, 'rb'))

        if self.generate_on_train:
            # divide boxes into each box and expand the annotations
            new_annotation = []
            for ann in self.annotation:
                phrases = ann['phrases']
                for phrase in phrases:
                    new_ann = ann.copy()
                    new_ann['phrases'] = phrase
                    new_annotation.append(new_ann)
            self.annotation = new_annotation
            if mask:
                print(f'In [Gen_on_mask] mode - Number of annotations: {len(self.annotation)}')
            else:
                print(f'In [Gen_on_train] mode - Number of annotations: {len(self.annotation)}')
        
        else:
            gt_format = {'info': {}, 'type': 'caption', 'annotations': [], 'images': [], 'licenses': []}
            gt_file = os.path.join(all_config.run_cfg['caption_gt_root'], f'{self.dataset_name}_{split}_gt.json')
            if not os.path.exists(gt_file):
                gts = [{'id': int(ann['img_id']), 'image_id': int(ann['img_id']), 'caption': ann['caption']} for ann in self.annotation]
                gts = sorted(gts, key = lambda x:x['id'])
                gt_format['annotations'] = gts
                gt_format['images'] = [{'id': int(ann['img_id'])} for ann in self.annotation]
                json.dump(gt_format, open(gt_file, 'w'))

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        img_id = int(ann['img_id'])

        if self.dataset_name == 'flickr30k_entities':
            image_path = os.path.join(self.base_path + 'flickr30k-images/', ann['img_id']+'.jpg')
        elif self.dataset_name == 'mscoco':
            image_path = os.path.join(self.base_path, ann['img_path'])

        ann_copy = None
        if self.generate_on_train:
            ann_copy = ann.copy()
        
        image = Image.open(image_path).convert('RGB')
        if self.generate_on_train:
            boxes = ann['phrases']['boxes']
            image = blacken_box(image.copy(), boxes)
        image = self.vis_processor(image)

        return image, img_id, ann_copy




