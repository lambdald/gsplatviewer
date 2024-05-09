from pathlib import Path
from plyfile import PlyData, PlyElement
import numpy as np
import torch
import math
import os
try:
    from tensordict import tensorclass
except:
    os.system('pip install tensordict==0.2.0')


@tensorclass
class GaussianModel:
    means: torch.Tensor
    opacities: torch.Tensor
    features: torch.Tensor
    scalings: torch.Tensor
    rotations: torch.Tensor

    @staticmethod
    def from_file(ply_filepath: Path) -> "GaussianModel":
        assert ply_filepath.exists()
        plydata = PlyData.read(str(ply_filepath))
        #! xyz
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )

        #! opacity
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        #! dc color
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))

        max_sh_degree = int(math.sqrt((len(extra_f_names) + 3) / 3)) - 1
        print(f'max sh degree={max_sh_degree}')

        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

        # [N, 3, d]
        features = np.concatenate([features_dc, features_extra], axis=-1)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        return GaussianModel(
            means=torch.from_numpy(xyz),
            opacities=torch.from_numpy(opacities),
            rotations=torch.from_numpy(rots),
            scalings=torch.from_numpy(scales),
            features=torch.from_numpy(features),
            batch_size=xyz.shape[:1],
        )

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC

        num_dc = self.features.shape[1]
        num_rest = (self.features.shape[2] - 1) * self.features.shape[1]

        for i in range(num_dc):
            l.append("f_dc_{}".format(i))
        for i in range(num_rest):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self.scalings.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self.rotations.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def to_file(self, ply_filepath: Path):

        ply_filepath.parent.mkdir(exist_ok=True, parents=True)

        xyz = self.means.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        f_dc = self.features[:, :, :1].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.features[:, :, 1:].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.opacities.detach().cpu().numpy()
        scale = self.scalings.detach().cpu().numpy()
        rotation = self.rotations.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(str(ply_filepath))

    def to_splat_file(self, splat_filepath: Path):

        splat_filepath.parent.mkdir(exist_ok=True, parents=True)

        sorted_indices = torch.argsort(-self.scalings.sum(dim=-1).exp() / (1 + torch.exp(-self.opacities.view(-1))))
        sorted_data: GaussianModel = self[sorted_indices]

        SH_C0 = 0.28209479177387814
        position = sorted_data.means
        scales = sorted_data.scalings.exp()
        color = 0.5 + sorted_data.features[..., 0] * SH_C0
        opacity = 1 / (1 + torch.exp(-sorted_data.opacities))
        color = torch.cat([color, opacity], dim=-1) * 255
        rots = torch.nn.functional.normalize(sorted_data.rotations, p=2, dim=-1) * 128 + 128

        n_pts = sorted_data.shape[0]

        part1 = torch.cat([position, scales], dim=-1).to(torch.float32).cpu().contiguous().numpy()
        part2 = torch.cat([color, rots], dim=-1).clip(0, 255).to(torch.uint8).cpu().contiguous().numpy()

        with splat_filepath.open("wb") as f:
            for idx in range(n_pts):
                f.write(part1[idx].tobytes())
                f.write(part2[idx].tobytes())
