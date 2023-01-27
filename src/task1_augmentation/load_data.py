from pathlib import Path
from random import random

import torch

# TODO: consider if numpy or torch types should replace float and int.
# TODO: handle corrupted .obj file with adequate exceptions.

MODEL_VERTEX = list[float, float, float, float, float, float]  # x y z R G B
MODEL_VERTEX_NORMAL = list[float, float, float]  # i j k
MODEL_TEXTURE_VERTEX = list[float, float, float]  # vt u v w
MODEL_FACE = list[float]  # v/vt/vn v/vt/vn v/vt/vn ...

VERTEX_START_LINE_START = "v"
VERTEX_NORMAL_LINE_START = "vn"
TEXTURE_START_LINE_START = "vt"
MODEL_FACE_LINE_START = "f"
COMMENT_LINE_START = "#"


class ObjFileContent:
    """
    Defines content of a single .obj file.

    .obj file consists of:

    v – model vertex (point in a 3D space)
    vn – vertex normal vector
    vt – coordinates of a texture vertex
    f – model face

    Each of these file elements have a field in this class.
    """
    original_file_name: str = ""
    model_vertex_list: list[MODEL_VERTEX]
    model_vertex_normal_list: list[MODEL_VERTEX_NORMAL]
    model_texture_vertex_list: list[MODEL_TEXTURE_VERTEX]
    model_face_list: list[MODEL_FACE]
    meta: list[str]

    def __init__(self, path: Path | None):
        self.model_vertex_list = []
        self.model_vertex_normal_list = []
        self.model_texture_vertex_list = []
        self.model_face_list = []
        self.meta = []

        if path:
            self.original_file_name = path.name

            with path.open(mode="r") as f:
                for line in f:
                    # VERTEX_START_LINE_START must be after VERTEX_NORMAL_LINE_START and TEXTURE_START_LINE_START

                    if line.startswith(VERTEX_NORMAL_LINE_START):
                        content_str = line[len(VERTEX_NORMAL_LINE_START) + 1:]
                        content = [float(c) for c in content_str.split(" ")]
                        self.model_vertex_normal_list.append(content)
                    elif line.startswith(TEXTURE_START_LINE_START):
                        content_str = line[len(TEXTURE_START_LINE_START) + 1:]
                        content = [float(c) for c in content_str.split(" ")]
                        self.model_texture_vertex_list.append(content)
                    elif line.startswith(VERTEX_START_LINE_START):
                        content_str = line[len(VERTEX_START_LINE_START) + 1:]
                        content = [float(c) for c in content_str.split(" ")]
                        self.model_vertex_list.append(content)
                    elif line.startswith(MODEL_FACE_LINE_START):
                        content_str = line[len(MODEL_FACE_LINE_START) + 1:]
                        content = [float(c) for c in content_str.split(" ")]
                        self.model_face_list.append(content)
                    elif line.startswith(COMMENT_LINE_START):
                        self.meta.append(line)
                    elif line == "\n":
                        continue
                    else:
                        raise RuntimeError(f"Corrupted .obj file. Problem found in line: {line}.")

    @staticmethod
    def from_batch(batch) -> list["ObjFileContent"]:
        obj_file_contents_list = []

        for row_idx in range(len(batch)):
            positions = batch[row_idx].pos
            colour = batch[row_idx].colour
            model_vertex_torch = torch.hstack([positions, colour])

            obj_content = ObjFileContent(path=None)
            obj_content.original_file_name = batch[row_idx].original_file_name
            obj_content.model_vertex_list = [[y.item() for y in x] for x in model_vertex_torch]
            obj_content.meta = batch[row_idx].meta_data
            obj_file_contents_list.append(obj_content)

        return obj_file_contents_list

    def to_file(self, path: Path):
        with path.open("w") as f:
            f.writelines(self.meta)  # Losing the sequence, but it does not matter for the file content
            for model_vertex_content in self.model_vertex_list:
                f.write(f"{VERTEX_START_LINE_START} {' '.join([str(x) for x in model_vertex_content])}\n")
            # There are other value types to consider for full .obj support.

    def add_to_directory(self, dir: Path):
        """
        Save file to directory. The file name is reusing the name of an original file it was loaded from.

        :param dir: Path to directory to write to
        """
        rand_str = str(random())[2:7]
        target_path = dir / self.original_file_name.replace(".obj", f"_{rand_str}.obj")
        self.to_file(target_path)

    def to_torch(self) -> torch.Tensor:
        """
        :return: tuple of np.array. First torch.Tensor represents the points positions,
        second represents the RGB colour, third keeps those two together.
        """
        positions_array = torch.Tensor(self.model_vertex_list)
        return positions_array


if __name__ == '__main__':
    o = ObjFileContent(Path("/home/artur/PycharmProjects/anything_world/data/TASK1/model_1.obj"))
    point_cloud = o.to_torch()
    print()
