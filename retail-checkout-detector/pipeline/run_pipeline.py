"""Live pipeline: Camera -> YOLO Detection -> Tracking -> Alerts -> Render -> Screen.

Detection and tracking are combined via Ultralytics' model.track(), which
runs YOLO detection and then ByteTrack on top of it, attaching a persistent
track ID to each detected box (result.boxes.id). Alerts here are simple
item counting: each new track ID seen increments a running count.
"""

import argparse

import cv2
from ultralytics import YOLO


def run(weights, source, conf, tracker):
    model = YOLO(weights)

    seen_ids = set()
    item_count = 0

    # Camera + YOLO Detection + Tracking, combined:
    results_gen = model.track(
        source=source,
        tracker=tracker,
        persist=True,
        stream=True,  # yields one frame at a time - keeps memory flat for long-running video
        conf=conf,
        verbose=False,
    )

    for result in results_gen:
        # ---- Alerts: log/count only new track IDs ----
        if result.boxes.id is not None:
            ids = result.boxes.id.int().tolist()
            for track_id in ids:
                if track_id not in seen_ids:
                    seen_ids.add(track_id)
                    item_count += 1
                    print(f"[LOG] New item seen - ID {track_id} | total count: {item_count}")

        # ---- Render ----
        annotated = result.plot()  # draws boxes + track IDs
        cv2.putText(
            annotated,
            f"Total items: {item_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        # ---- Screen ----
        cv2.imshow("Product Counter", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to trained YOLO weights (best.pt).")
    parser.add_argument("--source", default="0", help="Camera index (e.g. 0) or a video file/RTSP URL.")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    args = parser.parse_args()

    # allow numeric webcam indices passed as strings
    source = int(args.source) if args.source.isdigit() else args.source

    run(args.weights, source, args.conf, args.tracker)


if __name__ == "__main__":
    main()
