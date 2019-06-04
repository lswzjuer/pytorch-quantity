# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
import argparse
import os
from loguru import logger

import torch
from roadtensor.common.config import cfg
from roadtensor.components.modeling.models.detector import build_detection_model
from roadtensor.common.utils.checkpoint import DetectronCheckpointer
from roadtensor.common.utils.collect_env import collect_env_info
from roadtensor.common.utils.miscellaneous import mkdir
from roadtensor.common.utils.utils import get_dataset_type
from roadtensor.components.engine.export import export

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/roadtensor/roadtensor/common/configs/retinanet/retinanet_X_101_32x8d_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    # Load model
    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    # Load weight
    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)
    export_model_name = os.path.basename(checkpointer.get_checkpoint_file())
    if export_model_name == "":
        logger.info("There are no checkpoint, loading a random model")
        export_model_name = "random_model"
    export_model_name = export_model_name.split(".")[0] + (".onnx" if cfg.EXPORT.ONNX_ONLY else ".plan")

    # export the net
    if cfg.OUTPUT_DIR:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "export")
        mkdir(output_folder)
        export_model_path = os.path.join(output_folder, export_model_name)
    # Start export
    export(model, cfg, export_model_path)

if __name__ == "__main__":
    main()
