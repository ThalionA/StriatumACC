"""Frame reading and per-ROI motion energy for the top (face/body) camera."""

import json
import subprocess
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Roi:
    """Rectangle in pixel coordinates of the frame being processed:
    x, y = top-left corner (x right, y down), w, h = width, height."""

    x: int
    y: int
    w: int
    h: int

    def scaled(self, factor):
        return Roi(*(round(v * factor) for v in (self.x, self.y, self.w, self.h)))


def probe(path):
    """Width, height and decoded frame count (container fps is not trusted:
    the MVS camera writes 68 fps in the header, real rates are 25-33 fps)."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
           "-show_entries", "stream=width,height,nb_read_packets", "-of", "json", str(path)]
    s = json.loads(subprocess.run(cmd, capture_output=True, check=True, text=True).stdout)["streams"][0]
    return int(s["width"]), int(s["height"]), int(s["nb_read_packets"])


def iter_gray_frames(path, scale=1.0):
    """Yield uint8 grey frames decoded by ffmpeg, optionally downscaled."""
    w, h, _ = probe(path)
    ow, oh = round(w * scale), round(h * scale)
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={ow}:{oh}:flags=area,format=gray",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    nbytes = ow * oh
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            yield np.frombuffer(buf, np.uint8).reshape(oh, ow)
    finally:
        proc.stdout.close()
        proc.wait()


def frame_features(frames, rois, wheel_roi):
    """Per-frame features from ONE pass over the frames (each video is decoded once).

    Motion energy: mean absolute frame-to-frame difference inside each ROI in
    `rois` (grey levels). Wheel shift: translation of the texture inside
    `wheel_roi` between consecutive frames, by Hann-windowed phase correlation;
    dy > 0 = content moved DOWN the image (px/frame). response = correlation
    peak height (0-1), low when blur or occlusion leaves no trackable texture.
    Unlike motion energy the shift does not saturate with speed (until it nears
    half the ROI height, where it wraps).

    Returns (me: {name: array}, dy, dx, response); frame 0 is NaN throughout.
    """
    import cv2

    window = cv2.createHanningWindow((wheel_roi.w, wheel_roi.h), cv2.CV_64F)
    me = {name: [] for name in rois}
    dy, dx, resp = [], [], []
    prev = None
    for frame in frames:
        cur = frame.astype(np.int16)
        if prev is None:
            h, w = frame.shape
            for name, r in {**rois, "_wheel": wheel_roi}.items():
                if r.x < 0 or r.y < 0 or r.x + r.w > w or r.y + r.h > h:
                    raise ValueError(f"ROI {name}={r} outside {w}x{h} frame")
            for name in rois:
                me[name].append(np.nan)
            dy.append(np.nan)
            dx.append(np.nan)
            resp.append(np.nan)
        else:
            diff = np.abs(cur - prev)
            for name, r in rois.items():
                me[name].append(float(diff[r.y:r.y + r.h, r.x:r.x + r.w].mean()))
            wr = wheel_roi
            # astype() makes fresh copies: OpenCV 4.10's phaseCorrelate multiplies
            # its inputs by the window IN PLACE, so reused arrays get windowed twice.
            (sx, sy), rr = cv2.phaseCorrelate(
                prev[wr.y:wr.y + wr.h, wr.x:wr.x + wr.w].astype(np.float64),
                cur[wr.y:wr.y + wr.h, wr.x:wr.x + wr.w].astype(np.float64), window)
            dy.append(sy)
            dx.append(sx)
            resp.append(rr)
        prev = cur
    return {k: np.asarray(v) for k, v in me.items()}, np.asarray(dy), np.asarray(dx), np.asarray(resp)


def diff_maps_by_label(frames, labels, n_labels):
    """Sum of |frame_i - frame_(i-1)| images per label (1..n_labels; 0 = skip),
    and the frame count per label. Frame 0 has no predecessor and is skipped.
    Mean map for label k = sums[k-1] / counts[k-1]."""
    sums, counts, prev = None, np.zeros(n_labels, int), None
    for i, frame in enumerate(frames):
        cur = frame.astype(np.int16)
        if sums is None:
            sums = np.zeros((n_labels, *frame.shape))
        elif labels[i] > 0:
            sums[labels[i] - 1] += np.abs(cur - prev)
            counts[labels[i] - 1] += 1
        prev = cur
    return sums, counts
