
"""
The pipeline runner: every stage wired together and executed end to end.

This module owns the frame loop. Each individual stage lives in its own file
and is imported here, so this file describes the order of operations rather
than the operations themselves.
"""

# The os module is used to create the output directory and build file paths.
import os
# The YOLO class provides both detection and multi-object tracking.
from ultralytics import YOLO

# Paths, class names and thresholds.
import src.config as config
# Stage 1: reshape the raw model output into per-class tables.
from src.detection import get_frame_detections
# Stage 2: build the three kinds of relationship between detected objects.
from src.association import associate_riders_to_motorcycles, associate_heads_to_riders, associate_plates_to_motorcycles
# Stage 3: combine many frames into a single verdict.
from src.temporal import smooth_helmet_status, decide_violation
# Stage 4: the accumulating record held for each rider.
from src.rm_instance import RMInstanceManager
# Stage 5: render pipeline state into reviewable images.
from src.visualization import draw_rm_instances, crop_plate


# Process an entire video and return everything the run produced.
def run_pipeline(model_path=None, video_path=None, frame_limit=None, save_sample_frames=True):
    # Fall back to the configured weights when the caller supplies no path.
    model_path = model_path or config.MODEL_PATH
    # Fall back to the configured video when the caller supplies no path.
    video_path = video_path or config.VIDEO_PATH
    # Fall back to the configured frame cap, where None means the whole video.
    frame_limit = frame_limit if frame_limit is not None else config.FRAME_LIMIT

    # Load the trained detection weights into memory once.
    model = YOLO(model_path)

    # Begin tracking. stream=True yields one frame at a time rather than
    # accumulating every result, and persist=True carries tracking identifiers
    # forward between frames so a rider keeps the same ID throughout.
    results = model.track(
        source=video_path, classes=list(config.CLASS_NAMES.keys()),
        tracker=config.TRACKER, persist=True, stream=True, verbose=False,
    )

    # Holds one accumulating record per rider.
    manager = RMInstanceManager()
    # Maps each rider to the list of helmet observations recorded for them.
    observations = {}
    # Maps each motorcycle to the most recent number plate box matched to it.
    plate_matches = {}
    # Collects one annotated evidence image per confirmed violation.
    sample_frames = []

    # Counts how many frames were actually processed, for the final report.
    frame_count = 0
    # Process the video one frame at a time.
    for i, r in enumerate(results):
        # Stop early when a frame cap has been set and reached.
        if frame_limit is not None and i >= frame_limit:
            break
        # Frame numbering is one-based in the report, while i is zero-based.
        frame_count = i + 1

        # Step 1: reshape this frame's raw output into per-class tables.
        det, confs = get_frame_detections(r)

        # Step 2: pair riders with motorcycles and record each pairing as a vote.
        for rider_id, moto_id in associate_riders_to_motorcycles(det["rider"], det["motorcycle"]):
            manager.get_or_create(rider_id, moto_id)

        # Step 3: determine each rider's helmet status for this frame.
        head_matches = associate_heads_to_riders(
            det["rider"], det["helmet"], det["no_helmet"],
            helmet_confs=confs["helmet"], no_helmet_confs=confs["no_helmet"],
        )
        # Append the observation, building the history smoothing will later use.
        for rider_id, (status, model_conf) in head_matches.items():
            observations.setdefault(rider_id, []).append((status, model_conf))

        # Pair number plates with motorcycles and remember the plate's box.
        for moto_id, plate_id in associate_plates_to_motorcycles(det["motorcycle"], det["license_plate"]).items():
            plate_matches[moto_id] = det["license_plate"][plate_id]

        # Step 4: re-evaluate every rider's verdict from all evidence to date.
        for rider_id, inst in manager.instances_by_rider.items():
            # Summarise this rider's observations, which may return None.
            smoothed = smooth_helmet_status(observations, rider_id)
            # Store the majority status, or None when smoothing declined.
            inst.helmet_status = smoothed[0] if smoothed else None
            # Store the averaged detector confidence behind that status.
            inst.confidence = smoothed[1] if smoothed else None
            # Apply the confidence threshold to reach the final verdict.
            inst.is_violation = decide_violation(smoothed)
            # Attach the number plate of whichever motorcycle they are on.
            if inst.motorcycle_id in plate_matches:
                inst.plate_box = plate_matches[inst.motorcycle_id]

        # Step 5: capture one annotated snapshot per confirmed violation.
        if save_sample_frames:
            # Consider only riders actually visible in the current frame.
            for rider_id in det["rider"]:
                # The accumulated record for this rider, if one exists.
                inst = manager.instances_by_rider.get(rider_id)
                # Avoid producing a near-identical image on every later frame.
                already_captured = any(rid == rider_id for rid, _, _ in sample_frames)
                # Capture only on the first frame at which the verdict is True.
                if inst and inst.is_violation is True and not already_captured:
                    # Render the evidence view, highlighting this rider alone.
                    annotated = draw_rm_instances(r.orig_img, det, manager.instances_by_rider, highlight_rider_id=rider_id)
                    # Store the image alongside the rider and frame it came from.
                    sample_frames.append((rider_id, i, annotated))

    # Report what was processed and from where.
    print(f"Processed {frame_count} frames from: {video_path}")
    print(f"Model: {model_path}")
    # Report the aggregate counts the manager accumulated.
    for k, v in manager.summary().items():
        print(f"  {k}: {v}")

    # Return every artefact the run produced, for inspection by the caller.
    return manager, observations, plate_matches, sample_frames


# Write the text report and the evidence images to the output directory.
def save_outputs(manager, sample_frames, output_dir=None):
    # Fall back to the configured output location when none is supplied.
    output_dir = output_dir or config.OUTPUT_DIR
    # Create the directory if it does not already exist, without failing if it does.
    os.makedirs(output_dir, exist_ok=True)

    # Path of the plain-text summary of every record held.
    report_path = os.path.join(output_dir, "rm_instances_report.txt")
    # Write one line per record, using the readable form defined on RMInstance.
    with open(report_path, "w") as f:
        for inst in manager.all_instances():
            f.write(repr(inst) + "\n")
    # Confirm where the report was written.
    print(f"\nSaved report: {report_path}")

    # OpenCV is imported here because it is required only for image output.
    import cv2
    # Write one image file per captured violation, named for rider and frame.
    for rider_id, frame_num, img in sample_frames:
        img_path = os.path.join(output_dir, f"violation_rider{rider_id}_frame{frame_num}.png")
        cv2.imwrite(img_path, img)
    # Report the count only when at least one snapshot was produced.
    if sample_frames:
        print(f"Saved {len(sample_frames)} violation snapshot(s) to: {output_dir}")


# Executed only when this file is run directly, not when it is imported.
if __name__ == "__main__":
    # Run the full pipeline with every setting taken from the configuration.
    manager, observations, plate_matches, sample_frames = run_pipeline()
    # Write the report and the evidence images to disk.
    save_outputs(manager, sample_frames)
