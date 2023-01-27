import tempfile
from pathlib import Path
from typing import Union, List, Tuple

import pymeshlab
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from src import DATA_PATH
from src.task1_augmentation.load_data import ObjFileContent

TASK1_DATA_PATH = DATA_PATH / "TASK1"
TASK1_OUTPUT = DATA_PATH / "TASK1_OUTPUT"
TASK1_OUTPUT.mkdir(parents=True, exist_ok=True)


class AnythingWorldDataset(InMemoryDataset):
    POS_END_IDX = 3

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return [str(p) for p in (TASK1_DATA_PATH / "raw").glob("*.obj")]

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['all_data.pt']

    @property
    def obj_file_contents(self) -> List[ObjFileContent]:
        return [ObjFileContent(Path(raw_file_name)) for raw_file_name in self.raw_file_names]

    def download(self):
        raise NotImplementedError()

    def process(self):
        tensors_list = [obj_content.to_torch()
                        for obj_content in self.obj_file_contents]
        data_list = [Data(pos=tensor[:, :self.POS_END_IDX],
                          colour=tensor[:, self.POS_END_IDX:],
                          original_file_name=obj_file_content.original_file_name,
                          meta_data=obj_file_content.meta)
                     for tensor, obj_file_content in zip(tensors_list, self.obj_file_contents)]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def augment_with_meshlab(obj_file: ObjFileContent) -> ObjFileContent:
    """
    Performs augmenting using PyMeshLab. Some file operations here could be dropped.

    The augmenting itself is doing a reconstruction of the mesh based on the point cloud,
    followed by a sample from the mesh.
    """
    ms = pymeshlab.MeshSet()
    with tempfile.NamedTemporaryFile(suffix=".obj") as f1, tempfile.NamedTemporaryFile(suffix=".obj") as f2:
        obj_file.to_file(Path(f1.name))

        ms.load_new_mesh(str(f1.name))
        ms.generate_surface_reconstruction_ball_pivoting()
        ms.generate_sampling_stratified_triangle(samplenum=10000)
        ms.transfer_attributes_per_vertex(sourcemesh=0, targetmesh=1)
        ms.save_current_mesh(f2.name)

        output_obj = ObjFileContent(Path(f2.name))
        output_obj.original_file_name = obj_file.original_file_name

    return output_obj


if __name__ == '__main__':
    output_directory = Path()

    dataset = AnythingWorldDataset(root=str(TASK1_DATA_PATH))
    data_loader = DataLoader(dataset, batch_size=11, shuffle=False)

    transform = T.Compose([T.RandomJitter(0.1),
                           T.RandomFlip(0),
                           T.RandomShear(0.2),
                           T.RandomScale((0.95, 1.05)),
                           T.Center()])  # Fills the requirement to put it "roughly in the center"
    # TODO: original data appears to have min/max centering rather than global

    for batch in data_loader:
        batch_out = transform(batch)
        obj_content_list = ObjFileContent.from_batch(batch_out)
        for obj_content in obj_content_list:
            obj_content = augment_with_meshlab(obj_content)
            obj_content.add_to_directory(dir=TASK1_OUTPUT)
