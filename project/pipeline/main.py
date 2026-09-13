"""Main pipeline runner -- Stages 2-10, wired end to end."""

import os
from ultralytics import YOLO

import config
from detection import get_frame_detections
from association import associate_riders_to_motorcycles, associate_heads_to_riders, associate_plates_to_motorcycles
from temporal import smooth_helmet_status, decide_violation
from rm_instance import RMInstanceManager
from visualization import draw_rm_instances, crop_plate


def run_pipeline(model_path=None, video_path=None, frame_limit=None, save_sample_frames=True):
    model_path = model_path or config.MODEL_PATH
    video_path = video_path or config.VIDEO_PATH
    frame_limit = frame_limit if frame_limit is not None else config.FRAME_LIMIT

    model = YOLO(model_path)

    results = model.track(
        source=video_path, classes=list(config.CLASS_NAMES.keys()),
        tracker=config.TRACKER, persist=True, stream=True, verbose=False,
    )

    manager = RMInstanceManager()
    observations = {}
    plate_matches = {}
    sample_frames = []

    frame_count = 0
    for i, r in enumerate(results):
        if frame_limit is not None and i >= frame_limit:
            break
        frame_count = i + 1

        det, confs = get_frame_detections(r)

        for rider_id, moto_id in associate_riders_to_motorcycles(det["rider"], det["motorcycle"]):
            manager.get_or_create(rider_id, moto_id)

        head_matches = associate_heads_to_riders(
            det["rider"], det["helmet"], det["no_helmet"],
            helmet_confs=confs["helmet"], no_helmet_confs=confs["no_helmet"],
        )
        for rider_id, (status, model_conf) in head_matches.items():
            observations.setdefault(rider_id, []).append((status, model_conf))

        for moto_id, plate_id in associate_plates_to_motorcycles(det["motorcycle"], det["license_plate"]).items():
            plate_matches[moto_id] = det["license_plate"][plate_id]

        for rider_id, inst in manager.instances_by_rider.items():
            smoothed = smooth_helmet_status(observations, rider_id)
            inst.helmet_status = smoothed[0] if smoothed else None
            inst.confidence = smoothed[1] if smoothed else None
            inst.is_violation = decide_violation(smoothed)
            if inst.motorcycle_id in plate_matches:
                inst.plate_box = plate_matches[inst.motorcycle_id]

        if save_sample_frames:
            for rider_id in det["rider"]:
                inst = manager.instances_by_rider.get(rider_id)
                already_captured = any(rid == rider_id for rid, _, _ in sample_frames)
                if inst and inst.is_violation is True and not already_captured:
                    annotated = draw_rm_instances(r.orig_img, det, manager.instances_by_rider, highlight_rider_id=rider_id)
                    sample_frames.append((rider_id, i, annotated))

    print(f"Processed {frame_count} frames from: {video_path}")
    print(f"Model: {model_path}")
    for k, v in manager.summary().items():
        print(f"  {k}: {v}")

    return manager, observations, plate_matches, sample_frames


def save_outputs(manager, sample_frames, output_dir=None):
    output_dir = output_dir or config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "rm_instances_report.txt")
    with open(report_path, "w") as f:
        for inst in manager.all_instances():
            f.write(repr(inst) + "\n")
    print(f"\nSaved report: {report_path}")

    import cv2
    for rider_id, frame_num, img in sample_frames:
        img_path = os.path.join(output_dir, f"violation_rider{rider_id}_frame{frame_num}.png")
        cv2.imwrite(img_path, img)
    if sample_frames:
        print(f"Saved {len(sample_frames)} violation snapshot(s) to: {output_dir}")


if __name__ == "__main__":
    manager, observations, plate_matches, sample_frames = run_pipeline()
    save_outputs(manager, sample_frames)