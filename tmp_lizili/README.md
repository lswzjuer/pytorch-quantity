# Roadtensor

This project aims at providing the necessary building blocks for easily
creating detection and segmentation models using Pytorch.

## Highlights and Roadmap

- **PyTorch 1.1:**
- **data preprocessing in GPU:**
- **Mixed precision and distributed training:**
- **Sparse Convolution:**  
- **Inference optimziation for tensorRT, fabu-chips and FPGA:**  
- **CPU and GPU efficient:**
- **Suported task:**: lidar_fusion,vision2d, mono_depth, stereo_depth, parano_seg, traffic_light, traffic_lane, quantization, network prunning, online export.
- **Tensorboard:**
- **Remote vscode by [code-server](https://github.com/cdr/code-server):**

## Getting start

### Getting in docker

```bash
# Setup docker env
./docker/docker_start.sh

# Enter docker env
./docker/docker_into.sh
```

### Multi-GPU training

We use internally `torch.distributed.launch` in order to launch
multi-gpu training. This utility function from PyTorch spawns as many
Python processes as the number of GPUs we want to use, and each Python
process will only use a single GPU.

```bash
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS /path_to_lidar/tools/train_net.py --config-file "path/to/config/file.yaml"
```

### Perform training on coco dataset with RetinaNet

```bash
ln -s /private/liuzili/data/coco /datasets/coco
chmod +x train.sh
./train.sh
```

### Remote features

```bash
# check ports
cat ~/.roadtensor_port

# 1). Connect your docker directly from remote with the ssh port, for example
ssh -X -p 32768 xxx@192.168.3.xxx

# 2). Edit your code with the vscode port,
# open your browser and enter "localhost:port" or "your-ip:port"

# 3). Monitor your traning loss with the tensorboardX port
tensorboard --logdir=models/log
# check your localhost:port or your-ip:port
```

## Adding your own tasks

## Adding your own dataset

This implementation adds support for COCO-style datasets.
But adding support for training on a new dataset can be done as follows:

```python
from lidar.structures.bounding_box import BoxList

class MyDataset(object):
    def __init__(self, ...):
        # as you would do normally

    def __getitem__(self, idx):
        # load the image as a PIL Image
        image = ...

        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        boxes = [[0, 0, 10, 10], [10, 20, 50, 50]]
        # and labels
        labels = torch.tensor([10, 20])

        # create a BoxList from the boxes
        boxlist = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)

        if self.transforms:
            image, boxlist = self.transforms(image, boxlist)

        # return the image, the boxlist and the idx in your dataset
        return image, boxlist, idx

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        return {"height": img_height, "width": img_width}
```

That's it. You can also add extra fields to the boxlist, such as segmentation masks
(using `structures.segmentation_mask.SegmentationMask`), or even your own instance type.

## Contirbution

See [Contribution](http://git.fabu.ai/roadtensor/roadtensor/blob/master/contribution.md).
