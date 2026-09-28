import collections
import os

import cv2
import numpy as np
import open3d as o3d

SCANNETPP_IDS = []
SCANNETPP_LABELS = []


BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
BaseCamera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


class Image(BaseImage):
    @property
    def world_to_camera(self):
        R = qvec2rotmat(self.qvec)
        world2cam = np.eye(4)
        world2cam[:3, :3] = R
        world2cam[:3, 3] = self.tvec
        return world2cam


class Camera(BaseCamera):
    @property
    def K(self):
        K = np.eye(3)
        if self.model in {
            "SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL",
            "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE",
        }:
            K[0, 0] = self.params[0]
            K[1, 1] = self.params[0]
            K[0, 2] = self.params[1]
            K[1, 2] = self.params[2]
        elif self.model in {
            "PINHOLE", "OPENCV", "OPENCV_FISHEYE",
            "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE",
        }:
            K[0, 0] = self.params[0]
            K[1, 1] = self.params[1]
            K[0, 2] = self.params[2]
            K[1, 2] = self.params[3]
        else:
            raise NotImplementedError(f"Unsupported COLMAP camera model: {self.model}")
        return K


def read_images_text(path):
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) == 0 or line[0] == "#":
                continue

            elems = line.split()
            image_id = int(elems[0])
            qvec = np.array(tuple(map(float, elems[1:5])))
            tvec = np.array(tuple(map(float, elems[5:8])))
            camera_id = int(elems[8])
            image_name = elems[9]

            points_line = fid.readline().split()
            xys = np.column_stack([
                tuple(map(float, points_line[0::3])),
                tuple(map(float, points_line[1::3])),
            ])
            point3D_ids = np.array(tuple(map(int, points_line[2::3])))

            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def read_cameras_text(path):
    cameras = {}
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if len(line) == 0 or line[0] == "#":
                continue

            elems = line.split()
            camera_id = int(elems[0])
            cameras[camera_id] = Camera(
                id=camera_id,
                model=elems[1],
                width=int(elems[2]),
                height=int(elems[3]),
                params=np.array(tuple(map(float, elems[4:]))),
            )
    return cameras


def resolve_scene_root(data_root, seq_name):
    data_root = os.path.expanduser(data_root)
    if os.path.basename(os.path.normpath(data_root)) == seq_name:
        return data_root
    return os.path.join(data_root, seq_name)


class ScanNetPPDslrDataset:
    """
    Reader for ScanNet++-style DSLR exports laid out as:
    images/*.JPG, depths/*.png, sparse/0/{cameras,images}.txt, mesh_aligned_0.05.ply.
    """

    def __init__(self, args) -> None:
        self.seq_name = args.seq_name
        data_root = getattr(args, "scannetpp_dslr_root", "./data/scannetpp_dslr/data")
        self.root = resolve_scene_root(data_root, self.seq_name)

        self.rgb_dir = os.path.join(self.root, "images")
        self.depth_dir = os.path.join(self.root, "depths")
        self.colmap_dir = os.path.join(self.root, "sparse", "0")
        self.segmentation_dir = os.path.join(self.root, "output", "mask")
        self.object_dict_dir = os.path.join(self.root, "output", "object")

        mesh_name = getattr(args, "scannetpp_dslr_mesh_name", "mesh_aligned_0.05.ply")
        self.mesh_path = os.path.join(self.root, mesh_name)
        self.point_cloud_path = self.mesh_path
        self.depth_scale = float(getattr(args, "scannetpp_dslr_depth_scale", 1000.0))

        self.load_meta_data()

    def load_meta_data(self):
        cameras = read_cameras_text(os.path.join(self.colmap_dir, "cameras.txt"))
        images = read_images_text(os.path.join(self.colmap_dir, "images.txt"))

        frame_id_list = []
        rgb_paths = {}
        depth_paths = {}
        image_sizes = {}
        intrinsics = {}
        extrinsics = {}
        missing = []

        for image in sorted(images.values(), key=lambda item: item.id):
            frame_id = os.path.splitext(os.path.basename(image.name))[0]
            camera = cameras[image.camera_id]
            rgb_path = os.path.join(self.rgb_dir, image.name)
            depth_path = os.path.join(self.depth_dir, frame_id + ".png")

            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                missing.append((frame_id, rgb_path, depth_path))
                continue

            frame_id_list.append(frame_id)
            rgb_paths[frame_id] = rgb_path
            depth_paths[frame_id] = depth_path
            image_sizes[frame_id] = (camera.width, camera.height)
            intrinsics[frame_id] = camera.K
            extrinsics[frame_id] = np.linalg.inv(image.world_to_camera)

        if missing:
            examples = ", ".join(item[0] for item in missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} registered COLMAP frames are missing RGB/depth files. "
                f"Examples: {examples}"
            )
        if not frame_id_list:
            raise RuntimeError(f"No registered RGB-D frames found under {self.root}")

        self.frame_id_list = frame_id_list
        self.rgb_paths = rgb_paths
        self.depth_paths = depth_paths
        self.image_sizes = image_sizes
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.image_size = image_sizes[frame_id_list[0]]

    def get_frame_list(self, stride):
        return self.frame_id_list[::stride]

    def get_intrinsics(self, frame_id):
        width, height = self.image_sizes[frame_id]
        intrinsic_matrix = self.intrinsics[frame_id]

        intrinsic_cam_parameters = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_cam_parameters.set_intrinsics(
            width,
            height,
            intrinsic_matrix[0, 0],
            intrinsic_matrix[1, 1],
            intrinsic_matrix[0, 2],
            intrinsic_matrix[1, 2],
        )
        return intrinsic_cam_parameters

    def get_extrinsic(self, frame_id):
        return self.extrinsics[frame_id]

    def get_depth(self, frame_id):
        depth = cv2.imread(self.depth_paths[frame_id], cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Depth not found: {self.depth_paths[frame_id]}")
        depth = depth / self.depth_scale
        return depth.astype(np.float32)

    def get_rgb(self, frame_id, change_color=True):
        rgb = cv2.imread(self.rgb_paths[frame_id])
        if rgb is None:
            raise FileNotFoundError(f"RGB not found: {self.rgb_paths[frame_id]}")
        if change_color:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb

    def get_segmentation(self, frame_id, align_with_depth=False):
        segmentation_path = os.path.join(self.segmentation_dir, frame_id + ".png")
        if not os.path.exists(segmentation_path):
            assert False, f"Segmentation not found: {segmentation_path}"
        segmentation = cv2.imread(segmentation_path, cv2.IMREAD_UNCHANGED)
        if align_with_depth:
            width, height = self.image_sizes[frame_id]
            if segmentation.shape[:2] != (height, width):
                segmentation = cv2.resize(segmentation, (width, height), interpolation=cv2.INTER_NEAREST)
        return segmentation

    def get_frame_path(self, frame_id):
        return self.rgb_paths[frame_id], os.path.join(self.segmentation_dir, frame_id + ".png")

    def get_label_features(self):
        label_features_dict = np.load(f"data/text_features/scannetpp.npy", allow_pickle=True).item()
        return label_features_dict

    def get_scene_points(self):
        mesh = o3d.io.read_triangle_mesh(self.mesh_path)
        vertices = np.asarray(mesh.vertices)
        return vertices

    def get_label_id(self):
        self.class_id = SCANNETPP_IDS
        self.class_label = SCANNETPP_LABELS

        self.label2id = {}
        self.id2label = {}
        for label, id in zip(self.class_label, self.class_id):
            self.label2id[label] = id
            self.id2label[id] = label

        return self.label2id, self.id2label
