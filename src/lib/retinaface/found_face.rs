use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct FoundFace {
    pub(crate) bbox: [f32; 4],
    pub(crate) score: f32,
    pub(crate) landmarks: [[f32; 2]; 5],
}