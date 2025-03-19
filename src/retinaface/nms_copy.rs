// Largely inspired by lsnms: https://github.com/remydubois/lsnms

use std::cmp::Ordering;
// use crate::utils;
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use num_traits::{Num, ToPrimitive};
const ONE: f64 = 1.0;
const ZERO: f64 = 0.0;
pub fn min<N>(a: N, b: N) -> N
where
    N: Num + PartialOrd,
{
    if a < b {
        return a;
    } else {
        return b;
    }
}

pub fn max<N>(a: N, b: N) -> N
where
    N: Num + PartialOrd,
{
    if a > b {
        return a;
    } else {
        return b;
    }
}


#[inline(always)]
pub fn area<N>(bx: N, by: N, bxx: N, byy: N) -> N
where
    N: Num + PartialEq + PartialOrd + ToPrimitive,
{
    (bxx - bx) * (byy - by)
}

/// Performs non-maximum suppression (NMS) on a set of bounding boxes using their scores and IoU.
/// # Arguments
///
/// * `boxes` - A 2D array of shape `(num_boxes, 4)` representing the coordinates in xyxy format of the bounding boxes.
/// * `scores` - A 1D array of shape `(num_boxes,)` representing the scores of the bounding boxes.
/// * `iou_threshold` - A float representing the IoU threshold to use for filtering.
/// * `score_threshold` - A float representing the score threshold to use for filtering.
///
/// # Returns
///
/// A 1D array of shape `(num_boxes,)` representing the indices of the bounding boxes to keep.
///
/// # Examples
///
/// ```
/// use ndarray::{arr2, Array1};
/// use powerboxesrs::nms::nms;
///
/// let boxes = arr2(&[[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]]);
/// let scores = Array1::from(vec![1.0, 1.0]);
/// let keep = nms(&boxes, &scores, 0.8, 0.0);
/// assert_eq!(keep, vec![0, 1]);
/// ```
pub fn nms<'a, N, BA, SA>(
    boxes: BA,
    scores: SA,
    iou_threshold: f64,
    score_threshold: f64,
) -> Vec<usize>
where
    N: Num + PartialEq + PartialOrd + ToPrimitive + Copy + PartialEq + 'a,
    BA: Into<ArrayView2<'a, N>>,
    SA: Into<ArrayView1<'a, f64>>,
{
    let boxes = boxes.into();
    let scores = scores.into();
    assert_eq!(boxes.nrows(), scores.len_of(Axis(0)));

    let order: Vec<usize> = {
        let mut indices: Vec<_> = if score_threshold > ZERO {
            // filter out boxes lower than score threshold
            scores
                .iter()
                .enumerate()
                .filter(|(_, &score)| score >= score_threshold)
                .map(|(idx, _)| idx)
                .collect()
        } else {
            (0..scores.len()).collect()
        };
        // sort box indices by scores
        indices.sort_unstable_by(|&a, &b| {
            scores[b].partial_cmp(&scores[a]).unwrap_or(Ordering::Equal)
        });
        indices
    };

    let mut keep: Vec<usize> = Vec::new();
    let mut suppress = vec![false; order.len()];

    for (i, &idx) in order.iter().enumerate() {
        if suppress[i] {
            continue;
        }
        keep.push(idx);
        let box1 = boxes.row(idx);
        let b1x = box1[0];
        let b1y = box1[1];
        let b1xx = box1[2];
        let b1yy = box1[3];
        let area1 = powerboxesrs::nms::area(b1x, b1y, b1xx, b1yy);
        for j in (i + 1)..order.len() {
            if suppress[j] {
                continue;
            }
            let box2 = boxes.row(order[j]);
            let b2x = box2[0];
            let b2y = box2[1];
            let b2xx = box2[2];
            let b2yy = box2[3];

            // Intersection-over-union
            let x = max(b1x, b2x);
            let y = max(b1y, b2y);
            let xx = min(b1xx, b2xx);
            let yy = min(b1yy, b2yy);
            if x > xx || y > yy {
                // Boxes are not intersecting at all
                continue;
            };
            // Boxes are intersecting
            let intersection: N = powerboxesrs::nms::area(x, y, xx, yy);
            let area2: N = powerboxesrs::nms::area(b2x, b2y, b2xx, b2yy);
            let union: N = area1 + area2 - intersection;
            let iou: f64 = intersection.to_f64().unwrap() / union.to_f64().unwrap();
            if iou > iou_threshold {
                suppress[j] = true;
            }
        }
    }
    keep
}
