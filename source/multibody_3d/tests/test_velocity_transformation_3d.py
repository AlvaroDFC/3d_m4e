# tests/test_velocity_transformation_3d.py  (pytest)
from multibody_3d import JointSystem3D
from multibody_3d import VelocityTransformation3D
from multibody_3d.multibody_core.velocity_transformation_3d import demo_write_pairs_chain

def test_iter_write_pairs_root_to_leaf_chain():
    pairs = demo_write_pairs_chain()
    assert pairs == [
        (1, 0), (2, 0), (3, 0),
        (2, 1), (3, 1),
        (3, 2),
    ]