use std::path::PathBuf;
use anyhow::Error;
use ndarray::Array4;
use once_cell::sync::Lazy;
#[allow(unused_imports)]
use ort::{CPUExecutionProvider, CUDAExecutionProvider, DirectMLExecutionProvider, ExecutionProvider, GraphOptimizationLevel, OneDNNExecutionProvider, OpenVINOExecutionProvider, Session, TensorRTExecutionProvider};
use crate::project_dir;


use super::retinaface_resnet;
use super::retinaface_mobilenet;
pub use super::found_face;

const MOBILENET_ONNX:  Lazy<PathBuf> = Lazy::new(|| project_dir().join("src").join("retinaface").join("mobilenet_retinaface.onnx"));
const RESNET_ONNX: Lazy<PathBuf> = Lazy::new(|| project_dir().join("src").join("retinaface").join("resnet_retinaface.onnx"));


pub enum ModelKind {
    MobileNet,
    ResNet,
}

pub struct RetinaFaceFaceDetector {
    session: Session,
    model: ModelKind,
}


impl RetinaFaceFaceDetector {
    pub fn new(model_kind: ModelKind) -> RetinaFaceFaceDetector {
        // ort::init_from(r#"C:\Users\tomokazu\RustroverProjects\ameba_blog_downloader\onnxruntime-win-x64-gpu-1.17.3\lib\onnxruntime.dll"#).commit().unwrap();
        let execution_providers = [
            // OpenVINOExecutionProvider::default().build(),
            // OneDNNExecutionProvider::default().build(),
            TensorRTExecutionProvider::default().build(),
            CUDAExecutionProvider::default().build(),
            DirectMLExecutionProvider::default().build(),
            CPUExecutionProvider::default().build(),
        ];
        for execution_provider in &execution_providers {
            if execution_provider.is_available().unwrap() {
                println!("Selected Execution Provider: {}", execution_provider.as_str());
                break;
            }
        }
        let session_builder = Session::builder().unwrap()
            .with_execution_providers(execution_providers).unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
            .with_intra_threads(16).unwrap();

        match model_kind {
            ModelKind::MobileNet => {
                RetinaFaceFaceDetector {
                    session: session_builder.commit_from_file(MOBILENET_ONNX.as_path()).unwrap(),
                    model: model_kind,
                }
            }
            ModelKind::ResNet => {
                RetinaFaceFaceDetector {
                    session: session_builder.commit_from_file(RESNET_ONNX.as_path()).unwrap(),
                    model: model_kind,
                }
            }
        }
    }

    pub fn infer(&self, image: Vec<u8>) -> Vec<found_face::FoundFace> {
        match self.model {
            ModelKind::MobileNet => retinaface_mobilenet::infer(&self.session, image).unwrap(),
            ModelKind::ResNet => retinaface_resnet::infer(&self.session, image).unwrap()
        }
    }

    pub fn image_to_array(&self, image: Vec<u8>) -> Result<Array4<f32>, Error> {
        match self.model {
            ModelKind::MobileNet => retinaface_mobilenet::transform(image),
            ModelKind::ResNet => retinaface_resnet::transform(image)
        }
    }
}
