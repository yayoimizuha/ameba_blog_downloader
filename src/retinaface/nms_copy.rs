/// Non-Maximum Suppression (NMS) implementation.
///
/// Ported from the Python/NumPy reference:
/// <https://github.com/supernotman/RetinaFace_Pytorch/blob/master/utils.py>
///
/// The bounding box format is `[x1, y1, x2, y2]` (top-left and bottom-right corners).
use ndarray::{Array, Axis, Ix1, Ix2, s};

pub fn nms(boxes: &Array<f32, Ix2>, scores: &Array<f32, Ix1>, iou_threshold: f32) -> Vec<usize> {
    let x1 = boxes.slice(s![.., 0]).to_owned();
    let y1 = boxes.slice(s![.., 1]).to_owned();
    let x2 = boxes.slice(s![.., 2]).to_owned();
    let y2 = boxes.slice(s![.., 3]).to_owned();

    // Compute areas: (x2 - x1 + 1) * (y2 - y1 + 1)
    let areas = (&x2 - &x1 + 1.0) * (&y2 - &y1 + 1.0);

    // Sort by score in ascending order (we pop from the end → highest first)
    // This matches: order = np.argsort(score)
    let mut order = {
        let mut indices: Vec<usize> = (0..scores.len()).collect();
        indices.sort_by(|&a, &b| {
            scores[a]
                .partial_cmp(&scores[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    };

    let mut keep = Vec::new();

    while !order.is_empty() {
        // Pick the index with the largest score (last element after ascending sort)
        let i = *order.last().unwrap();
        keep.push(i);

        // Compare against all remaining candidates (excluding the picked one)
        let rest = &order[..order.len() - 1];
        if rest.is_empty() {
            break;
        }

        let rest_indices: Vec<usize> = rest.to_vec();

        // Compute intersection coordinates
        let xx1 = x1.select(Axis(0), &rest_indices).mapv(|v| v.max(x1[i]));
        let yy1 = y1.select(Axis(0), &rest_indices).mapv(|v| v.max(y1[i]));
        let xx2 = x2.select(Axis(0), &rest_indices).mapv(|v| v.min(x2[i]));
        let yy2 = y2.select(Axis(0), &rest_indices).mapv(|v| v.min(y2[i]));

        // Compute intersection area: w = max(0, xx2 - xx1 + 1), h = max(0, yy2 - yy1 + 1)
        let w = (&xx2 - &xx1 + 1.0).mapv(|v| v.max(0.0));
        let h = (&yy2 - &yy1 + 1.0).mapv(|v| v.max(0.0));
        let intersection = &w * &h;

        // Compute IoU: intersection / (area_i + area_rest - intersection)
        let rest_areas = areas.select(Axis(0), &rest_indices);
        let ratio = &intersection / &(rest_areas + areas[i] - &intersection);

        // Keep only indices where ratio < iou_threshold
        // This matches: left = np.where(ratio < iou_threshold); order = order[left]
        order = rest_indices
            .iter()
            .zip(ratio.iter())
            .filter(|(_, &r)| r < iou_threshold)
            .map(|(&idx, _)| idx)
            .collect();
    }

    keep
}
