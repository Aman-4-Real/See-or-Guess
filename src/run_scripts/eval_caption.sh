
###
 # @Author: Aman
 # @Date: 2023-10-19 10:51:58
 # @Contact: cq335955781@gmail.com
 # @LastEditors: Aman
 # @LastEditTime: 2024-08-28 21:28:08
### 

### evaluation on the factual images (using flickr30k dataset as an example here)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 \
    ../evaluate.py --cfg-path ../lavis/run_cfgs/caption_flk_eval.yaml


### generate_on_train
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 \
#     ../evaluate.py --cfg-path ../lavis/run_cfgs/caption_eval_gen_on_train.yaml


### evaluation on the counterfactual images
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 \
#     ../evaluate.py --cfg-path ../lavis/run_cfgs/caption_coco_eval_mask_gen.yaml


