"""Top-camera ROIs shared by every video driver."""

from striatum_video.motion import Roi

# Full-resolution (500 x 600) pixel coordinates, chosen on frames from 1201 and
# 1206 (same rig geometry). Wheel = textured surface right of the spout block,
# clear of the mouse's shadow; mouth = spout tip and jaw; whiskers = snout and pad.
# mouth and whiskers carry mostly RUNNING motion (partial r with licks | speed <= 0.13).
ROIS = {
    "wheel": Roi(380, 320, 110, 270),
    "mouth": Roi(270, 220, 90, 80),
    "whiskers": Roi(200, 150, 200, 80),
    # Lick spout block + tube. Chosen from speed-matched lick-minus-lick-free
    # difference maps (1201, 1206; figures/lick_maps.png): the spout jolts when
    # licked and running barely moves it. 1105/1106 were NOT used to place it.
    "spout": Roi(250, 295, 120, 230),
}

# Face region for motion SVD (motion_svd.py): head, snout, whisker pads, mouth,
# forepaws and the spout tip, from below the head bar (y ~110) to the top of the
# spout block (y ~310). Its left/right edges include some wheel, so running
# will appear as a component; that is handled by the VR-speed base model.
FACE_SVD_ROI = Roi(160, 110, 280, 200)
FACE_SVD_BIN = 5  # 5 x 5 pixel blocks -> 56 x 40 = 2240 motion features
