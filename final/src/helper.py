"""
Visualisation helpers used by the notebook.

These functions exist purely to keep plotting code out of the notebook. They
are not part of the detection pipeline, and nothing in the pipeline imports
them. Class names are passed in as an argument rather than imported, so this
module carries no dependency on the configuration.
"""

# OpenCV is used only to convert frames from blue-green-red to red-green-blue.
import cv2
# Matplotlib renders the panels and every annotation drawn on them.
import matplotlib.pyplot as plt
# Patches supplies the rectangle used to outline each detected object.
import matplotlib.patches as patches

# Compact class labels, because the full names collide in a dense frame.
SHORT_NAMES = {
    "motorcycle": "moto",
    "rider": "rider",
    "helmet": "helm",
    "no_helmet": "no-helm",
    "license_plate": "plate",
}


# Reduce one result object to two lookups, both keyed by tracking identifier.
def by_track(r, class_names):
    """Return (box_of, name_of), mapping each tracking identifier to its box
    corners and to its readable class name. Both share the same keys."""
    # A frame in which the tracker found nothing yields None rather than a list.
    if r.boxes.id is None:
        return {}, {}
    # Tracking identifiers, converted from tensor values to plain integers.
    ids = [int(t) for t in r.boxes.id.tolist()]
    # Corner coordinates of each box, in the same order as the identifiers.
    box_of = dict(zip(ids, r.boxes.xyxy.tolist()))
    # Readable class name of each box, translated from its numeric class.
    name_of = dict(zip(ids, [class_names[int(c)] for c in r.boxes.cls.tolist()]))
    # Both lookups share the same keys, so either may drive a drawing loop.
    return box_of, name_of


# Draw two consecutive frames together to make tracking visible.
def show_tracked_pair(frame_a, frame_b, class_names, pad=70, layout="auto",
                      width=18, figsize=None):
    """Colour each object by its tracking identifier, so one object keeps one
    colour in both panels. Objects present in only one frame are drawn dashed.

    pad is the margin in pixels added around the region containing every box.

    layout accepts "stacked", "side", or "auto". Traffic detections usually
    occupy a wide, short strip of the frame, which reads better stacked, so
    "auto" selects the arrangement that matches the shape of that strip.

    width is the figure width in inches. The height is derived from the crop
    so the panels are never letterboxed. Passing figsize overrides both.
    """
    # Reduce each frame to its per-track lookups.
    b1, n1 = by_track(frame_a, class_names)
    b2, n2 = by_track(frame_b, class_names)

    # Identifiers present in both frames, which is precisely what tracking supplies.
    common = sorted(set(b1) & set(b2))
    # Identifiers the tracker reported in only one of the two frames.
    only_1, only_2 = sorted(set(b1) - set(b2)), sorted(set(b2) - set(b1))

    # Report the composition of the two frames before drawing them.
    print(f"frame 1: {len(b1)} objects   |   frame 2: {len(b2)} objects")
    print(f"tracked through both frames ({len(common)}): {common}")
    print(f"present only in frame 1:     {only_1}")
    print(f"present only in frame 2:     {only_2}")

    # # List what each identifier refers to, since on-image labels are abbreviated.
    # print("\nidentifier -> class")
    # for tid in sorted(set(b1) | set(b2)):
    #     # Mark whether the tracker held this object across both frames.
    #     held = "both frames" if tid in common else "one frame only"
    #     print(f"  #{tid:<3} {(n1.get(tid) or n2.get(tid)):<14} {held}")

    # Nothing can be drawn when neither frame contains a tracked object.
    all_boxes = list(b1.values()) + list(b2.values())
    if not all_boxes:
        print("\nno tracked objects in either frame")
        return

    # Both panels are cropped to the identical region. Using one shared crop is
    # essential, because two different crops would make coordinates incomparable.
    cx1 = max(0, int(min(b[0] for b in all_boxes)) - pad)
    cx2 = min(frame_a.orig_shape[1], int(max(b[2] for b in all_boxes)) + pad)
    cy1 = max(0, int(min(b[1] for b in all_boxes)) - pad)
    cy2 = min(frame_a.orig_shape[0], int(max(b[3] for b in all_boxes)) + pad)

    # Size the figure from the proportions of the crop itself. A fixed figure
    # size letterboxes the panels, because imshow preserves the image aspect
    # ratio and fills whatever space remains with blank margin.
    # max(1, ...) guards against a degenerate crop of no area, which would
    # otherwise divide by zero when the aspect ratio is computed.
    crop_w, crop_h = max(1, cx2 - cx1), max(1, cy2 - cy1)
    # Width divided by height, so a value above 1 describes a wide, short crop.
    aspect = crop_w / crop_h

    # A wide, short crop reads better stacked, because each panel then spans
    # the full figure width rather than half of it.
    if layout == "auto":
        layout = "stacked" if aspect > 2.0 else "side"
    # Grid shape, and the width in inches available to one panel.
    nrows, ncols = (2, 1) if layout == "stacked" else (1, 2)
    panel_w = width if layout == "stacked" else width / 2
    # The panel height that matches the image exactly, leaving no blank margin.
    panel_h = panel_w / aspect
    # Add a strip for the caption above and for each panel title.
    figsize = figsize or (width, nrows * (panel_h + 0.55) + 0.5)

    # Two panels arranged so one object can be compared across the two frames.
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Draw each frame into its own panel using identical logic.
    for ax, (title, r, box_of, name_of) in zip(
            axes, [("Frame 1", frame_a, b1, n1), ("Frame 2", frame_b, b2, n2)]):
        # Display the cropped frame in the channel order matplotlib expects.
        ax.imshow(cv2.cvtColor(r.orig_img[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2RGB))
        # Outline and label every object the tracker reported in this frame.
        for tid, box in box_of.items():
            # Shift the box into crop coordinates, as the panel begins at (cx1, cy1).
            x1, y1, x2, y2 = box[0]-cx1, box[1]-cy1, box[2]-cx1, box[3]-cy1
            # Colour is derived from the tracking identifier, so a given object
            # keeps one colour in both panels. This is what makes tracking visible.
            color = plt.cm.tab20(tid % 20)
            # An object absent from the other frame is drawn dashed to mark it out.
            style = "solid" if tid in common else "dashed"
            # Outline the object itself.
            ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                                           edgecolor=color, lw=1.8, linestyle=style))
            # A dark backing plate keeps the label readable where boxes overlap.
            ax.text(x1, y1-3, f"{SHORT_NAMES.get(name_of[tid], name_of[tid])} #{tid}",
                    color=color, fontsize=6, weight="bold", va="bottom",
                    bbox=dict(facecolor="black", alpha=0.55, pad=0.6, edgecolor="none"))
        # Name the panel and state how many objects it contains.
        ax.set_title(f"{title}: {len(box_of)} objects", fontsize=12)
        # Pixel axis ticks carry no meaning here, so they are removed.
        ax.axis("off")

    # State plainly what the reader should take from the two panels.
    fig.suptitle("Identical colour and identical number indicate the same tracked object. "
                 "A dashed outline appears in only one of the two frames.", fontsize=12)
    # Reserve roughly half an inch at the top for the caption. The allowance is
    # absolute rather than a fraction, because a fraction of a tall figure
    # would leave a conspicuous blank band above the first panel.
    plt.tight_layout(rect=[0, 0, 1, 1 - 0.45 / figsize[1]])
    plt.show()


# Annotate one object in both frames to explain the four values stored in xyxy.
def show_box_anatomy(frame_a, frame_b, class_names, track_id=None, zoom_pad=95,
                     figsize=(13, 7)):
    """Mark the two corners a box is stored as, and report how they moved
    between the frames.

    track_id selects the object to examine. When it is None, the object
    tracked through both frames with the largest box in frame one is used,
    because a large box carries the annotations most legibly.
    """
    # Reduce each frame to its per-track lookups.
    b1, n1 = by_track(frame_a, class_names)
    b2, _ = by_track(frame_b, class_names)

    # Only objects present in both frames can be compared across them.
    common = sorted(set(b1) & set(b2))
    # Without a common object there is nothing to annotate.
    if not common:
        print("no object was tracked through both frames")
        return

    # Fall back to the largest common object when the caller names none.
    if track_id is None:
        track_id = max(common, key=lambda t: (b1[t][2]-b1[t][0]) * (b1[t][3]-b1[t][1]))
    # A caller-supplied identifier must actually appear in both frames.
    elif track_id not in common:
        print(f"identifier {track_id} is not present in both frames; choose from {common}")
        return

    # Report how the four corner values changed between the two frames.
    print(f"tracking identifier {track_id}, class '{n1[track_id]}'\n")
    print(f"{'':<10}{'x1':>8}{'y1':>8}{'x2':>8}{'y2':>8}{'width':>9}{'height':>9}")
    # One row per frame, showing the raw corner values and the size they imply.
    for label, box in (("frame 1", b1[track_id]), ("frame 2", b2[track_id])):
        x1, y1, x2, y2 = box
        print(f"{label:<10}{x1:>8.0f}{y1:>8.0f}{x2:>8.0f}{y2:>8.0f}{x2-x1:>9.0f}{y2-y1:>9.0f}")
    # Movement of each corner between the two frames, in pixels. These values are
    # small, which is what allows the tracker to recognise the same object.
    print(f"{'change':<10}" + "".join(f"{b2[track_id][k]-b1[track_id][k]:>+8.0f}" for k in range(4)))

    # One panel per frame, placed side by side for direct comparison.
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Annotate the same object in each of the two frames.
    for ax, (title, r, box) in zip(axes, [("Frame 1", frame_a, b1[track_id]),
                                          ("Frame 2", frame_b, b2[track_id])]):
        # The four values stored in xyxy, in the order they are stored.
        x1, y1, x2, y2 = box
        # Crop bounds around this object, clamped to the edges of the frame.
        zx1, zx2 = max(0, int(x1)-zoom_pad), min(r.orig_shape[1], int(x2)+zoom_pad)
        zy1, zy2 = max(0, int(y1)-zoom_pad), min(r.orig_shape[0], int(y2)+zoom_pad)
        # Display the cropped region for this frame.
        ax.imshow(cv2.cvtColor(r.orig_img[zy1:zy2, zx1:zx2], cv2.COLOR_BGR2RGB))
        # The same corner values expressed relative to the crop, for drawing.
        rx1, ry1, rx2, ry2 = x1-zx1, y1-zy1, x2-zx1, y2-zy1
        # Outline the object using the two corners the model reported.
        ax.add_patch(patches.Rectangle((rx1, ry1), rx2-rx1, ry2-ry1,
                                       fill=False, edgecolor="yellow", lw=2))
        # Mark the first pair of values, which is the TOP-LEFT corner.
        ax.plot(rx1, ry1, "o", color="red", ms=9)
        ax.annotate(f"(x1, y1) = ({x1:.0f}, {y1:.0f})", (rx1, ry1), textcoords="offset points",
                    xytext=(0, 12), ha="left", color="white", fontsize=9, weight="bold",
                    bbox=dict(facecolor="red", alpha=0.85, pad=1.5, edgecolor="none"))
        # Mark the second pair of values, which is the BOTTOM-RIGHT corner.
        ax.plot(rx2, ry2, "o", color="deepskyblue", ms=9)
        ax.annotate(f"(x2, y2) = ({x2:.0f}, {y2:.0f})", (rx2, ry2), textcoords="offset points",
                    xytext=(0, -16), ha="right", color="black", fontsize=9, weight="bold",
                    bbox=dict(facecolor="deepskyblue", alpha=0.9, pad=1.5, edgecolor="none"))
        # A horizontal span showing that the width is x2 minus x1.
        ax.annotate("", (rx1, ry2+32), (rx2, ry2+32), arrowprops=dict(arrowstyle="<->", color="white"))
        ax.text((rx1+rx2)/2, ry2+52, f"width = x2 - x1 = {x2-x1:.0f} px", color="white",
                ha="center", va="top", fontsize=9,
                bbox=dict(facecolor="black", alpha=0.5, pad=1.5, edgecolor="none"))
        # A vertical span showing that the height is y2 minus y1.
        ax.annotate("", (rx2+32, ry1), (rx2+32, ry2), arrowprops=dict(arrowstyle="<->", color="white"))
        # The height label is rotated so it fits within the narrow right margin.
        ax.text(rx2+48, (ry1+ry2)/2, f"height = y2 - y1 = {y2-y1:.0f} px", color="white",
                va="center", ha="center", rotation=90, fontsize=9,
                bbox=dict(facecolor="black", alpha=0.5, pad=1.5, edgecolor="none"))
        # Name the panel using the class and the tracking identifier.
        ax.set_title(f"{title}: {n1[track_id]} #{track_id}", fontsize=12)
        # Pixel axis ticks carry no meaning here, so they are removed.
        ax.axis("off")

    # Record the coordinate convention, which is a frequent source of confusion.
    fig.suptitle("xyxy stores two corners: (x1, y1) top-left and (x2, y2) bottom-right. "
                 "In image coordinates y increases DOWNWARDS, so y1 is above y2.", fontsize=11)
    # Reserve a strip at the top for the caption so it does not overlap the panels.
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()
