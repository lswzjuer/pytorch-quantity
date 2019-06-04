import os
import numpy as np
import pickle
import torch
import torch.utils.data
import fire
import tqdm

from PIL import Image
from pathlib import Path
from loguru import logger

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits

def _get_available_scenes(nusc):
    available_scenes = []
    print("total scene num:", len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
            if not sd_rec['next'] == "":
                sd_rec = nusc.get('sample_data', sd_rec['next'])
            else:
                has_more_frames = False
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print("exist scene num:", len(available_scenes))
    return available_scenes


def _fill_trainval_infos(nusc,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10):
    trainnusc_infos = []
    valnusc_infos = []
    from pyquaternion import Quaternion
    for sample in tqdm.tqdm(nusc.sample):
        lidar_token = sample["data"]["LIDAR_TOP"]
        cam_front_token = sample["data"]["CAM_FRONT"]
        sd_rec = nusc.get('sample_data', sample['data']["LIDAR_TOP"])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_front_token)
        assert Path(lidar_path).exists(
        ), "you must download all trainval data."
        info = {
            "lidar_path": lidar_path,
            "cam_front_path": cam_path,
            "token": sample["token"],
            "sweeps": [],
            "lidar2ego_translation": cs_record['translation'],
            "lidar2ego_rotation": cs_record['rotation'],
            "ego2global_translation": pose_record['translation'],
            "ego2global_rotation": pose_record['rotation'],
            "timestamp": sample["timestamp"],
        }

        l2e_r = info["lidar2ego_rotation"]
        l2e_t = info["lidar2ego_translation"]
        e2g_r = info["ego2global_rotation"]
        e2g_t = info["ego2global_translation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        sd_rec = nusc.get('sample_data', sample['data']["LIDAR_TOP"])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == "":
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
                cs_record = nusc.get('calibrated_sensor',
                                     sd_rec['calibrated_sensor_token'])
                pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
                lidar_path = nusc.get_sample_data_path(sd_rec['token'])
                sweep = {
                    "lidar_path": lidar_path,
                    "sample_data_token": sd_rec['token'],
                    "lidar2ego_translation": cs_record['translation'],
                    "lidar2ego_rotation": cs_record['rotation'],
                    "ego2global_translation": pose_record['translation'],
                    "ego2global_rotation": pose_record['rotation'],
                    "timestamp": sd_rec["timestamp"]
                }
                l2e_r_s = sweep["lidar2ego_rotation"]
                l2e_t_s = sweep["lidar2ego_translation"]
                e2g_r_s = sweep["ego2global_rotation"]
                e2g_t_s = sweep["ego2global_translation"]
                # sweep->ego->global->ego'->lidar
                l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
                e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

                R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
                    np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
                    np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T) + l2e_t @ np.linalg.inv(l2e_r_mat).T
                sweep["sweep2lidar_rotation"] = R.T  # points @ R.T + T
                sweep["sweep2lidar_translation"] = T
                sweeps.append(sweep)
            else:
                break
        info["sweeps"] = sweeps
        if not test:
            annotations = [nusc.get('sample_annotation', token)
                           for token in sample['anns']]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)
            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NusceneDataset.NameMapping:
                    names[i] = NusceneDataset.NameMapping[names[i]]
            names = np.array(names)
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
            assert len(gt_boxes) == len(annotations), f"{len(gt_boxes)}, {len(annotations)}"
            info["gt_boxes"] = gt_boxes
            info["gt_names"] = names
            info["num_lidar_pts"] = np.array(
                [a["num_lidar_pts"] for a in annotations])
            info["num_radar_pts"] = np.array(
                [a["num_lidar_pts"] for a in annotations])
        if sample["scene_token"] in train_scenes:
            trainnusc_infos.append(info)
        else:
            valnusc_infos.append(info)
    return trainnusc_infos, valnusc_infos


def load_nuscene_dataset(dataroot="/private/xieliang/data/nuscenes/v1.0-mini",
                         version="v1.0-mini",
                         target_dir=Path("."),
                         max_sweeps=10):
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

    if version == "v1.0-trainval":
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == "v1.0-test":
        train_scenes = splits.test
        val_scenes = []
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError("unknown")
    available_scenes = _get_available_scenes(nusc)
    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(
        filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]["token"]
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]["token"]
        for s in val_scenes
    ])

    trainnusc_infos, valnusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, max_sweeps=max_sweeps)
    logger.info(
        f"train sample: {len(trainnusc_infos)}, val sample: {len(valnusc_infos)}"
    )
    metadata = {
        "version": version,
    }
    data = {
        "infos": trainnusc_infos,
        "metadata": metadata,
    }

    # Save it to pickle
    with open(target_dir / "infos_train.pkl", 'wb') as f:
        pickle.dump(data, f)
    data["infos"] = valnusc_infos
    with open(target_dir / "infos_val.pkl", 'wb') as f:
        pickle.dump(data, f)


class NusceneDataset(torch.utils.data.Dataset):
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }

    def __init__(self,
                 root_path,
                 info_path,
                 class_names=None,
                 transforms=None):
        self.root_path = Path(root_path)
        with open(info_path, 'rb') as f:
            data = pickle.load(f)
        self.nusc_infos = data["infos"]
        self.metadata = data["metadata"]
        self.class_names = class_names
        self.transforms = transforms
        # kitti map: nusc det name -> kitti eval name
        self._kitti_name_mapping = {
            "car": "car",
            "pedestrian": "pedestrian",
        }  # we only eval these classes in kitti
        self.version = self.metadata["version"]
        self.eval_version = "cvpr_2019"

    def __len__(self):
        return len(self.nusc_infos)

    def __getitem__(self, idx):
        input_dict = self.get_sensor_data(idx)
        example = self.transforms(input_dict=input_dict)
        example["metadata"] = input_dict["metadata"]
        if "anchors_mask" in example:
            example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
        return example

    def get_sensor_data(self, query):
        idx = query
        read_test_image = False
        if isinstance(query, dict):
            assert "lidar" in query
            idx = query["lidar"]["idx"]
            read_test_image = "cam" in query

        info = self.nusc_infos[idx]
        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
            },
            "metadata": {
                "token": info["token"]
            },
        }
        lidar_path = Path(info['lidar_path'])
        points = np.fromfile(
            str(lidar_path), dtype=np.float32, count=-1).reshape([-1,
                                                                  5])
        points[:, 3] /= 255
        points[:, 4] = 0
        sweep_points_list = [points]
        ts = info["timestamp"] / 1e6

        for sweep in info["sweeps"]:
            points_sweep = np.fromfile(
                str(sweep["lidar_path"]), dtype=np.float32,
                count=-1).reshape([-1, 5])
            sweep_ts = sweep["timestamp"] / 1e6
            points_sweep[:, 3] /= 255
            points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                "sweep2lidar_rotation"].T
            points_sweep[:, :3] += sweep["sweep2lidar_translation"]
            points_sweep[:, 4] = ts - sweep_ts
            sweep_points_list.append(points_sweep)

        points = np.concatenate(sweep_points_list, axis=0)[:, [0, 1, 2, 4]]

        if read_test_image:
            if Path(info["cam_front_path"]).exists():
                with open(str(info["cam_front_path"]), 'rb') as f:
                    image_str = f.read()
            else:
                image_str = None
            res["cam"] = {
                "type": "camera",
                "data": image_str,
                "datatype": Path(info["cam_front_path"]).suffix[1:],
            }
        res["lidar"]["points"] = points
        if 'gt_boxes' in info:
            res["lidar"]["annotations"] = {
                'boxes': info["gt_boxes"],
                'names': info["gt_names"],
            }
        return res


if __name__ == '__main__':
    # export dataset pickle
    fire.Fire(load_nuscene_dataset)

    # test nuscene dataset loader
    dataroot = "/private/xieliang/data/nuscenes/v1.0-mini"
    version = "v1.0-mini"
    dataset = NusceneDataset(dataroot, "infos_train.pkl")

