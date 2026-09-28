from pathlib import Path

from utils.instance_teacher_adapter import find_instance_teacher_ref, is_instance_level_teacher_dir


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def find_region_teacher_path(region_dir, image_stem, instance_teacher_map_mode="auto"):
    """Find the instance-level teacher reference for one image."""
    if not region_dir:
        return None

    region_dir = Path(region_dir)
    if is_instance_level_teacher_dir(region_dir):
        return find_instance_teacher_ref(region_dir, image_stem, map_mode=instance_teacher_map_mode)

    raise RuntimeError(f"Expected an instance-level teacher directory: {region_dir}")
