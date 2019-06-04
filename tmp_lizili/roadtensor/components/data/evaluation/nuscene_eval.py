import numpy as np

def ground_truth_annotations(dataset):
    if "gt_boxes" not in dataset.nusc_infos[0]:
        return None
    from nuscenes.eval.detection.config import eval_detection_configs
    cls_range_map = eval_detection_configs[dataset.eval_version]["class_range"]
    gt_annos = []
    for info in dataset.nusc_infos:
        gt_names = info["gt_names"]
        gt_boxes = info["gt_boxes"]
        num_lidar_pts = info["num_lidar_pts"]
        mask = num_lidar_pts > 0
        gt_names = gt_names[mask]
        gt_boxes = gt_boxes[mask]
        num_lidar_pts = num_lidar_pts[mask]

        mask = np.array([n in dataset.kitti_name_mapping for n in gt_names],
                        dtype=np.bool_)
        gt_names = gt_names[mask]
        gt_boxes = gt_boxes[mask]
        num_lidar_pts = num_lidar_pts[mask]
        gt_names_mapped = [dataset.kitti_name_mapping[n] for n in gt_names]
        det_range = np.array([cls_range_map[n] for n in gt_names_mapped])
        det_range = det_range[..., np.newaxis] @ np.array([[-1, -1, 1, 1]])
        mask = (gt_boxes[:, :2] >= det_range[:, :2]).all(1)
        mask &= (gt_boxes[:, :2] <= det_range[:, 2:]).all(1)
        gt_names = gt_names[mask]
        gt_boxes = gt_boxes[mask]
        num_lidar_pts = num_lidar_pts[mask]
        # use occluded to control easy/moderate/hard in kitti
        easy_mask = num_lidar_pts > 15
        moderate_mask = num_lidar_pts > 7
        occluded = np.zeros([num_lidar_pts.shape[0]])
        occluded[:] = 2
        occluded[moderate_mask] = 1
        occluded[easy_mask] = 0
        N = len(gt_boxes)
        gt_annos.append({
            "bbox":
            np.tile(np.array([[0, 0, 50, 50]]), [N, 1]),
            "alpha":
            np.full(N, -10),
            "occluded":
            occluded,
            "truncated":
            np.zeros(N),
            "name":
            gt_names,
            "location":
            gt_boxes[:, :3],
            "dimensions":
            gt_boxes[:, 3:6],
            "rotation_y":
            gt_boxes[:, 6],
        })
    return gt_annos

def evaluation_kitti(dataset, predictions, output_folder):
    """eval by kitti evaluation tool.
    I use num_lidar_pts to set easy, mod, hard.
    easy: num>15, mod: num>7, hard: num>0.
    """
    print("++++++++NuScenes KITTI unofficial Evaluation:")
    print("++++++++easy: num_lidar_pts>15, mod: num_lidar_pts>7, hard: num_lidar_pts>0")
    print("++++++++The bbox AP is invalid. Don't forget to ignore it.")
    class_names = dataset.class_names
    gt_annos = ground_truth_annotations(dataset)
    if gt_annos is None:
        return None
    gt_annos = deepcopy(gt_annos)
    predictions = deepcopy(predictions)
    dt_annos = []
    for det in predictions:
        final_box_preds = det["box3d_lidar"].detach().cpu().numpy()
        label_preds = det["label_preds"].detach().cpu().numpy()
        scores = det["scores"].detach().cpu().numpy()
        anno = kitti.get_start_result_anno()
        num_example = 0
        box3d_lidar = final_box_preds
        for j in range(box3d_lidar.shape[0]):
            anno["bbox"].append(np.array([0, 0, 50, 50]))
            anno["alpha"].append(-10)
            anno["dimensions"].append(box3d_lidar[j, 3:6])
            anno["location"].append(box3d_lidar[j, :3])
            anno["rotation_y"].append(box3d_lidar[j, 6])
            anno["name"].append(class_names[int(label_preds[j])])
            anno["truncated"].append(0.0)
            anno["occluded"].append(0)
            anno["score"].append(scores[j])
            num_example += 1
        if num_example != 0:
            anno = {n: np.stack(v) for n, v in anno.items()}
            dt_annos.append(anno)
        else:
            dt_annos.append(kitti.empty_result_anno())
        num_example = dt_annos[-1]["name"].shape[0]
        dt_annos[-1]["metadata"] = det["metadata"]

    for anno in gt_annos:
        names = anno["name"].tolist()
        mapped_names = []
        for n in names:
            if n in dataset.NameMapping:
                mapped_names.append(dataset.NameMapping[n])
            else:
                mapped_names.append(n)
        anno["name"] = np.array(mapped_names)
    for anno in dt_annos:
        names = anno["name"].tolist()
        mapped_names = []
        for n in names:
            if n in dataset.NameMapping:
                mapped_names.append(dataset.NameMapping[n])
            else:
                mapped_names.append(n)
        anno["name"] = np.array(mapped_names)
    mappedclass_names = []
    for n in dataset.class_names:
        if n in dataset.NameMapping:
            mappedclass_names.append(dataset.NameMapping[n])
        else:
            mappedclass_names.append(n)

    z_axis = 2
    z_center = 0.5
    # for regular raw lidar data, z_axis = 2, z_center = 0.5.
    result_official_dict = get_official_eval_result(
        gt_annos,
        dt_annos,
        mappedclass_names,
        z_axis=z_axis,
        z_center=z_center)
    result_coco = get_coco_eval_result(
        gt_annos,
        dt_annos,
        mappedclass_names,
        z_axis=z_axis,
        z_center=z_center)
    return {
        "results": {
            "official": result_official_dict["result"],
            "coco": result_coco["result"],
        },
        "detail": {
            "official": result_official_dict["detail"],
            "coco": result_coco["detail"],
        },
    }

def evaluation_nusc(dataset, predictions, output_folder):
    version = dataset.version
    eval_set_map = {
        "v1.0-mini": "mini_val",
        "v1.0-trainval": "val",
    }
    gt_annos = dataset.ground_truth_annotations
    if gt_annos is None:
        return None
    nusc_annos = {}
    mappedclass_names = dataset.class_names
    token2info = {}
    for info in dataset.nusc_infos:
        token2info[info["token"]] = info
    for det in predictions:
        annos = []
        boxes = _second_det_to_nusc_box(det)
        boxes = _lidar_nusc_box_to_global(token2info[det["metadata"]["token"]], boxes)
        for i, box in enumerate(boxes):
            name = mappedclass_names[box.label]
            nusc_anno = {
                "sample_token": det["metadata"]["token"],
                "translation": box.center.tolist(),
                "size": box.wlh.tolist(),
                "rotation": box.orientation.elements.tolist(),
                "velocity": [0.0, 0.0],
                "detection_name": name,
                "detection_score": box.score,
                "attribute_name": '',
            }
            annos.append(nusc_anno)
        nusc_annos[det["metadata"]["token"]] = annos
    res_path = str(Path(output_folder) / "results_nusc.json")
    with open(res_path, "w") as f:
        json.dump(nusc_annos, f)
    eval_main_file = Path(__file__).resolve().parent / "nusc_eval.py"
    # why add \"{}\"? to support path with spaces.
    cmd = f"python {str(eval_main_file)} --root_path=\"{str(dataset.root_path)}\""
    cmd += f" --version={dataset.version} --eval_version={dataset.eval_version}"
    cmd += f" --res_path=\"{res_path}\" --eval_set={eval_set_map[dataset.version]}"
    cmd += f" --output_folder=\"{output_folder}\""
    # use subprocess can release all nusc memory after evaluation
    subprocess.check_output(cmd, shell=True)
    with open(Path(output_folder) / "metrics.json", "r") as f:
        metrics = json.load(f)
    detail = {}
    result = f"Nusc {version} Evaluation\n"
    for name in mappedclass_names:
        detail[name] = {}
        for k, v in metrics["label_aps"][name].items():
            detail[name][f"dist@{k}"] = v
        threshs = ', '.join(list(metrics["label_aps"][name].keys()))
        scores = list(metrics["label_aps"][name].values())
        scores = ', '.join([f"{s * 100:.2f}" for s in scores])
        result += f"{name} Nusc dist AP@{threshs}\n"
        result += scores
        result += "\n"
    return {
        "results": {
            "nusc": result
        },
        "detail": {
            "nusc": detail
        },
    }

def nuscene_evaluation(dataset, predictions, output_folder):
    res_kitti = evaluation_kitti(dataset, predictions, output_folder)

    res_nusc = evaluation_nusc(dataset, predictions, output_folder)
    res = {
        "results": {
            "nusc": res_nusc["results"]["nusc"],
            "kitti.official": res_kitti["results"]["official"],
            "kitti.coco": res_kitti["results"]["coco"],
        },
        "detail": {
            "eval.nusc": res_nusc["detail"]["nusc"],
            "eval.kitti": {
                "official": res_kitti["detail"]["official"],
                "coco": res_kitti["detail"]["coco"],
            },
        },
    }
    return res