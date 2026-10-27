
import numpy as np


def node_frame_index(nodes, history, pose_rows, start_position):
    if not history:
        return {}
    positions = {int(row['step']): np.asarray(row['xyz_m'], dtype=float)
                 for row in pose_rows}
    available = [int(record['id']) for record in history if int(record['id']) in positions]
    origin = np.asarray(start_position, dtype=float)
    result = {}
    for node in nodes:
        if not available:
            break
        anchor = node.get('anchor') or {}
        frame = anchor.get('frame_id')
        if frame in available:
            result[str(node['id'])] = int(frame)
            continue
        target = np.asarray(node['position'], dtype=float) - origin
        result[str(node['id'])] = min(
            available, key=lambda fid: float(np.linalg.norm(
                (positions[fid] - target)[[0, 2]])))
    return result
