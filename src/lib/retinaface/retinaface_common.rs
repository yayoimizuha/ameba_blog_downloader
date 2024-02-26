use ort::{CPUExecutionProvider, CUDAExecutionProvider, DirectMLExecutionProvider, ExecutionProvider, GraphOptimizationLevel, Session, TensorRTExecutionProvider};
use serde::{Deserialize, Serialize};

const MOBILENET_ONNX: &str = r#"C:\Users\tomokazu\RustroverProjects\ameba_blog_downloader\src\bin\retinaface_mobilenet.rs"#;
const RESNET_ONNX: &str = r#"C:\Users\tomokazu\RustroverProjects\ameba_blog_downloader\src\bin\retinaface_resnet.rs"#;


enum modelKind {
    MobileNet,
    ResNet,
}

pub struct RetinaFaceFaceDetector {
    session: Session,
    model: modelKind,
}

mod retinaface_resnet;
mod retinaface_mobilenet;
mod found_face;

impl RetinaFaceFaceDetector {
    pub fn new(model_kind: modelKind) -> RetinaFaceFaceDetector {
        let execution_providers = [
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
            modelKind::MobileNet => {
                RetinaFaceFaceDetector {
                    session: session_builder.with_model_from_file(MOBILENET_ONNX).unwrap(),
                    model: model_kind,
                }
            }
            modelKind::ResNet => {
                RetinaFaceFaceDetector {
                    session: session_builder.with_model_from_file(RESNET_ONNX).unwrap()
                    model: model_kind,
                }
            }
        }
    }
    pub fn infer(&self, image: Vec<u8>) -> Vec<found_face::FoundFace> {
        match self.model {
            modelKind::MobileNet => {
                retinaface_mobilenet::infer(&self.session, image).unwrap()
            }
            modelKind::ResNet => {
                retinaface_resnet::infer(&self.session, image).unwrap()
            }
        }
    }
}
