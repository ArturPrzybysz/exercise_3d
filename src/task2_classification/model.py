from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from src.task1_augmentation.load_data import ObjFileContent


def is_right_angle(points: np.array, tolerance: float = 1e-8) -> bool:
    """
    Use Pythagorean theory to test if a triangle created over 3 points has a 90-degree angle.

    :param points: 3x3 array, shape = (POINT_ID, POINT_POSITION)
    :param tolerance: the allowed difference between tested squared side lengths.

    :return: True if is right angle, False otherwise.
    """
    p0 = points[0]
    p1 = points[1]
    p2 = points[2]

    dist_01 = np.sqrt(np.sum(np.square(p0 - p1)))
    dist_02 = np.sqrt(np.sum(np.square(p0 - p2)))
    dist_12 = np.sqrt(np.sum(np.square(p1 - p2)))

    dists = list(sorted([dist_01, dist_02, dist_12]))
    a = dists[2] ** 2
    b = dists[0] ** 2 + dists[1] ** 2
    # If the angle is 90-deg, then the longest side of the triangle squared equals to the sum of other squared sides.

    return np.isclose(a, b, atol=tolerance)


def is_cuboid(obj_file_path: Path) -> bool:
    """
    Test whether .obj file contains a cuboid. The test is based on the knowledge that cuboid has 12 right angles.

    :param obj_file_path:
    :return:
    """
    obj_content = ObjFileContent(obj_file_path)
    vertex_list = obj_content.model_vertex_list
    vertex_np = np.array(vertex_list)
    if len(vertex_np) != 8:
        return False

    hull = ConvexHull(vertex_np)

    return all(is_right_angle(vertex_np[s]) for s in hull.simplices)


if __name__ == '__main__':
    path_cuboids = Path("/home/artur/PycharmProjects/anything_world/data/TASK2/cuboid/")
    path_polyhedron = Path("/home/artur/PycharmProjects/anything_world/data/TASK2/polyhedron")

    answers1 = [is_cuboid(p) is True for p in path_cuboids.glob("*.obj")]
    answers2 = [is_cuboid(p) is False for p in path_polyhedron.glob("*.obj")]

    assert sum(answers1) == len(answers1)
    assert sum(answers2) == len(answers2)

    print("All the .obj files have been classified correctly.")
