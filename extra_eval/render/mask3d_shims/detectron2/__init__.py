"""Minimal stand-in for the three detectron2 functions Mask3D imports (detectron2 itself needs a
CUDA build). Re-implemented after detectron2 (Apache-2.0): ``utils.comm.get_world_size`` and
``projects.point_rend.point_features.{point_sample, get_uncertain_point_coords_with_randomness}``.
In ``mask3d_infer.py`` they only feed Mask3D's validation loss, computed alongside the
predictions in ``test_step``; the exported proposals do not depend on them.
"""
