###
 # @Author: Aman
 # @Date: 2023-10-19 10:51:58
 # @Contact: cq335955781@gmail.com
 # @LastEditors: Aman
 # @LastEditTime: 2024-08-28 21:30:42
### 

### training the initial model
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 \
    ../train.py --cfg-path ../lavis/run_cfgs/caption_coco_ft.yaml


### for te loss
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 \
#     ../train.py --cfg-path ../lavis/run_cfgs/caption_coco_ft_te0999.yaml


### for nde loss
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 \
#     ../train.py --cfg-path ../lavis/run_cfgs/caption_coco_ft_nde0999.yaml



