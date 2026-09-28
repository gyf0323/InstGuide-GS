import math

import torch


_INT64_MAX = torch.iinfo(torch.int64).max
_TRIANGLE_ID_MASK = (1 << 32) - 1


def _project_vertices(vertices, camera):
    device = camera.full_proj_transform.device
    vertices = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    ones = torch.ones((vertices.shape[0], 1), dtype=torch.float32, device=device)
    vertices_h = torch.cat((vertices, ones), dim=1)

    clip = vertices_h @ camera.full_proj_transform.to(device=device, dtype=torch.float32)
    view = vertices_h @ camera.world_view_transform.to(device=device, dtype=torch.float32)
    ndc = clip[:, :3] / clip[:, 3:4].clamp_min(1e-8)

    width = int(camera.image_width)
    height = int(camera.image_height)
    # Match diff-gaussian-rasterization's ndc2Pix convention. The repository
    # camera matrices already use image-space y orientation, so no y flip.
    pixel_x = ((ndc[:, 0] + 1.0) * width - 1.0) * 0.5
    pixel_y = ((ndc[:, 1] + 1.0) * height - 1.0) * 0.5
    return pixel_x, pixel_y, view[:, 2]


def _pack_depth_and_triangle(depth, triangle_ids):
    depth = depth.to(dtype=torch.float32).contiguous()
    depth_bits = depth.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    triangle_ids = triangle_ids.to(torch.int64)
    if triangle_ids.numel() and int(triangle_ids.max().item()) > _TRIANGLE_ID_MASK:
        raise RuntimeError("GT mesh has more triangles than the rasterizer's 32-bit triangle id supports.")
    return (depth_bits << 32) | triangle_ids


def _rasterize_pair_batch(
    key_buffer,
    pair_triangle_ids,
    pair_tile_ids,
    triangles,
    pixel_x,
    pixel_y,
    view_z,
    width,
    height,
    tiles_x,
    tile_size,
    local_x,
    local_y,
    barycentric_epsilon,
):
    triangle_vertices = triangles[pair_triangle_ids]
    x0 = pixel_x[triangle_vertices[:, 0]].unsqueeze(1)
    x1 = pixel_x[triangle_vertices[:, 1]].unsqueeze(1)
    x2 = pixel_x[triangle_vertices[:, 2]].unsqueeze(1)
    y0 = pixel_y[triangle_vertices[:, 0]].unsqueeze(1)
    y1 = pixel_y[triangle_vertices[:, 1]].unsqueeze(1)
    y2 = pixel_y[triangle_vertices[:, 2]].unsqueeze(1)
    z0 = view_z[triangle_vertices[:, 0]].unsqueeze(1)
    z1 = view_z[triangle_vertices[:, 1]].unsqueeze(1)
    z2 = view_z[triangle_vertices[:, 2]].unsqueeze(1)

    tile_x = pair_tile_ids.remainder(tiles_x)
    tile_y = torch.div(pair_tile_ids, tiles_x, rounding_mode="floor")
    sample_x = tile_x.unsqueeze(1) * tile_size + local_x.unsqueeze(0)
    sample_y = tile_y.unsqueeze(1) * tile_size + local_y.unsqueeze(0)
    in_image = (sample_x < width) & (sample_y < height)
    sample_xf = sample_x.to(torch.float32)
    sample_yf = sample_y.to(torch.float32)

    denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    nondegenerate = denominator.abs() > 1e-12
    safe_denominator = torch.where(nondegenerate, denominator, torch.ones_like(denominator))
    bary0 = ((y1 - y2) * (sample_xf - x2) + (x2 - x1) * (sample_yf - y2)) / safe_denominator
    bary1 = ((y2 - y0) * (sample_xf - x2) + (x0 - x2) * (sample_yf - y2)) / safe_denominator
    bary2 = 1.0 - bary0 - bary1
    inside = (
        in_image
        & nondegenerate
        & (bary0 >= -barycentric_epsilon)
        & (bary1 >= -barycentric_epsilon)
        & (bary2 >= -barycentric_epsilon)
    )
    if not bool(inside.any().item()):
        return

    inverse_z = bary0 / z0 + bary1 / z1 + bary2 / z2
    depth = inverse_z.reciprocal()
    inside &= torch.isfinite(depth) & (depth > 0)
    if not bool(inside.any().item()):
        return

    pixel_indices = (sample_y * width + sample_x)[inside].long()
    candidate_triangles = pair_triangle_ids.unsqueeze(1).expand_as(sample_x)[inside]
    packed = _pack_depth_and_triangle(depth[inside], candidate_triangles)
    key_buffer.scatter_reduce_(
        0,
        pixel_indices,
        packed,
        reduce="amin",
        include_self=True,
    )


def _rasterize_triangle_subset(
    key_buffer,
    triangle_ids,
    triangles,
    pixel_x,
    pixel_y,
    view_z,
    width,
    height,
    znear,
    zfar,
    tile_size,
    pair_chunk_size,
    max_face_tile_pairs,
    local_x,
    local_y,
    barycentric_epsilon,
):
    if triangle_ids.numel() == 0:
        return 0

    triangle_vertices = triangles[triangle_ids]
    triangle_x = pixel_x[triangle_vertices]
    triangle_y = pixel_y[triangle_vertices]
    triangle_z = view_z[triangle_vertices]

    finite = torch.isfinite(triangle_x).all(dim=1) & torch.isfinite(triangle_y).all(dim=1)
    in_depth = (triangle_z > float(znear)).all(dim=1) & (triangle_z < float(zfar)).all(dim=1)
    min_x = torch.ceil(triangle_x.min(dim=1).values).long()
    max_x = torch.floor(triangle_x.max(dim=1).values).long()
    min_y = torch.ceil(triangle_y.min(dim=1).values).long()
    max_y = torch.floor(triangle_y.max(dim=1).values).long()
    intersects = (max_x >= 0) & (max_y >= 0) & (min_x < width) & (min_y < height)
    valid = finite & in_depth & intersects
    if not bool(valid.any().item()):
        return 0

    triangle_ids = triangle_ids[valid]
    min_x = min_x[valid].clamp(0, width - 1)
    max_x = max_x[valid].clamp(0, width - 1)
    min_y = min_y[valid].clamp(0, height - 1)
    max_y = max_y[valid].clamp(0, height - 1)
    valid_extent = (max_x >= min_x) & (max_y >= min_y)
    triangle_ids = triangle_ids[valid_extent]
    min_x = min_x[valid_extent]
    max_x = max_x[valid_extent]
    min_y = min_y[valid_extent]
    max_y = max_y[valid_extent]
    if triangle_ids.numel() == 0:
        return 0

    tile_min_x = torch.div(min_x, tile_size, rounding_mode="floor")
    tile_max_x = torch.div(max_x, tile_size, rounding_mode="floor")
    tile_min_y = torch.div(min_y, tile_size, rounding_mode="floor")
    tile_max_y = torch.div(max_y, tile_size, rounding_mode="floor")
    tile_widths = tile_max_x - tile_min_x + 1
    tile_counts = tile_widths * (tile_max_y - tile_min_y + 1)
    total_pairs = int(tile_counts.sum().item())

    if total_pairs > int(max_face_tile_pairs) and triangle_ids.numel() > 1:
        midpoint = triangle_ids.numel() // 2
        first_count = _rasterize_triangle_subset(
            key_buffer,
            triangle_ids[:midpoint],
            triangles,
            pixel_x,
            pixel_y,
            view_z,
            width,
            height,
            znear,
            zfar,
            tile_size,
            pair_chunk_size,
            max_face_tile_pairs,
            local_x,
            local_y,
            barycentric_epsilon,
        )
        second_count = _rasterize_triangle_subset(
            key_buffer,
            triangle_ids[midpoint:],
            triangles,
            pixel_x,
            pixel_y,
            view_z,
            width,
            height,
            znear,
            zfar,
            tile_size,
            pair_chunk_size,
            max_face_tile_pairs,
            local_x,
            local_y,
            barycentric_epsilon,
        )
        return first_count + second_count

    starts = torch.cumsum(tile_counts, dim=0) - tile_counts
    pair_triangle_ids = torch.repeat_interleave(triangle_ids, tile_counts)
    pair_starts = torch.repeat_interleave(starts, tile_counts)
    pair_tile_widths = torch.repeat_interleave(tile_widths, tile_counts)
    pair_tile_min_x = torch.repeat_interleave(tile_min_x, tile_counts)
    pair_tile_min_y = torch.repeat_interleave(tile_min_y, tile_counts)
    local_offsets = torch.arange(total_pairs, device=triangle_ids.device) - pair_starts
    pair_tile_x = pair_tile_min_x + local_offsets.remainder(pair_tile_widths)
    pair_tile_y = pair_tile_min_y + torch.div(local_offsets, pair_tile_widths, rounding_mode="floor")
    tiles_x = int(math.ceil(width / tile_size))
    pair_tile_ids = pair_tile_y * tiles_x + pair_tile_x

    for start in range(0, total_pairs, int(pair_chunk_size)):
        end = min(start + int(pair_chunk_size), total_pairs)
        _rasterize_pair_batch(
            key_buffer,
            pair_triangle_ids[start:end],
            pair_tile_ids[start:end],
            triangles,
            pixel_x,
            pixel_y,
            view_z,
            width,
            height,
            tiles_x,
            tile_size,
            local_x,
            local_y,
            barycentric_epsilon,
        )
    return int(triangle_ids.numel())


@torch.no_grad()
def rasterize_semantic_mesh(
    vertices,
    triangles,
    triangle_labels,
    camera,
    tile_size=16,
    face_chunk_size=200000,
    pair_chunk_size=2048,
    max_face_tile_pairs=2000000,
    barycentric_epsilon=1e-5,
):
    if int(tile_size) <= 0:
        raise ValueError("tile_size must be positive.")
    if int(face_chunk_size) <= 0 or int(pair_chunk_size) <= 0 or int(max_face_tile_pairs) <= 0:
        raise ValueError("Rasterizer chunk sizes must be positive.")

    device = camera.full_proj_transform.device
    vertices = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    triangles = torch.as_tensor(triangles, dtype=torch.long, device=device)
    triangle_labels = torch.as_tensor(triangle_labels, dtype=torch.long, device=device).view(-1)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise RuntimeError(f"Expected triangle indices shaped [F, 3], got {tuple(triangles.shape)}.")
    if triangle_labels.numel() != triangles.shape[0]:
        raise RuntimeError("Triangle label count does not match triangle count.")
    if triangles.numel() and (int(triangles.min().item()) < 0 or int(triangles.max().item()) >= vertices.shape[0]):
        raise RuntimeError("GT mesh triangle index is outside the vertex array.")

    width = int(camera.image_width)
    height = int(camera.image_height)
    pixel_x, pixel_y, view_z = _project_vertices(vertices, camera)
    key_buffer = torch.full((height * width,), _INT64_MAX, dtype=torch.int64, device=device)

    local_y, local_x = torch.meshgrid(
        torch.arange(int(tile_size), device=device, dtype=torch.long),
        torch.arange(int(tile_size), device=device, dtype=torch.long),
        indexing="ij",
    )
    local_x = local_x.reshape(-1)
    local_y = local_y.reshape(-1)

    processed_triangles = 0
    for start in range(0, triangles.shape[0], int(face_chunk_size)):
        end = min(start + int(face_chunk_size), triangles.shape[0])
        triangle_ids = torch.arange(start, end, dtype=torch.long, device=device)
        processed_triangles += _rasterize_triangle_subset(
            key_buffer,
            triangle_ids,
            triangles,
            pixel_x,
            pixel_y,
            view_z,
            width,
            height,
            float(camera.znear),
            float(camera.zfar),
            int(tile_size),
            int(pair_chunk_size),
            int(max_face_tile_pairs),
            local_x,
            local_y,
            float(barycentric_epsilon),
        )

    label_map = torch.full((height * width,), -1, dtype=torch.long, device=device)
    covered = key_buffer != _INT64_MAX
    if bool(covered.any().item()):
        winning_triangles = key_buffer[covered] & _TRIANGLE_ID_MASK
        label_map[covered] = triangle_labels[winning_triangles.long()]
    label_map = label_map.view(height, width)
    stats = {
        "numTriangles": int(triangles.shape[0]),
        "numProjectedTriangles": int(processed_triangles),
        "numCoveredPixels": int(covered.sum().item()),
        "numValidLabelPixels": int((label_map >= 0).sum().item()),
    }
    return label_map, stats
