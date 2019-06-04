# @Author: Guodong Xu
# @Date:   2019-01-25 17:34:55
# @Last Modified by:   Guodong Xu
# @Last Modified time: 2019-05-15 14:09:29
export PYTHONDONTWRITEBYTECODE=1
export NGPUS=1
export CUDA_VISIBLE_DEVICES=0
# python -m torch.distributed.launch --nproc_per_node=$NGPUS /roadtensor/roadtensor/components/tools/detection/train_net.py --config-file "roadtensor/common/configs/e2e_faster_rcnn_R_101_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 10 SOLVER.BASE_LR 0.0125 SOLVER.MAX_ITER 144000 SOLVER.STEPS "(96000, 128000)" TEST.IMS_PER_BATCH 10
# python -m torch.distributed.launch --nproc_per_node=$NGPUS /roadtensor/roadtensor/components/tools/detection/train_net.py --config-file "roadtensor/vision2d/configs/e2e_faster_rcnn_R_50_C4_1x.yaml"

# For training
python -m torch.distributed.launch --master_port 5999 --nproc_per_node=$NGPUS /roadtensor/roadtensor/components/tools/detection/train_net.py --config-file "roadtensor/components/configs/retinanet/retinanet_R-50-FPN_1x.yaml"
# For testing
# python -m torch.distributed.launch --master_port 5999 --nproc_per_node=$NGPUS /roadtensor/roadtensor/components/tools/detection/test_net.py --config-file "roadtensor/components/configs/retinanet/retinanet_R-50-FPN_1x.yaml" TEST.IMS_PER_BATCH 6
# For exporting
 # python /roadtensor/roadtensor/components/tools/detection/export_net.py --config-file "roadtensor/components/configs/retinanet/retinanet_R-50-FPN_1x.yaml"
 # python /roadtensor/roadtensor/components/tools/detection/export_net.py --config-file "roadtensor/components/configs/rpn_R_50_C4_1x.yaml"
 # python /roadtensor/roadtensor/components/tools/detection/export_net.py --config-file "roadtensor/components/configs/e2e_faster_rcnn_R_101_FPN_1x.yaml"