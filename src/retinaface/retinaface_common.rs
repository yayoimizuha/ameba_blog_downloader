use image::DynamicImage;
use ndarray::Array4;
use ort::{CPUExecutionProvider, CUDAExecutionProvider, DirectMLExecutionProvider, ExecutionProvider, GraphOptimizationLevel, OpenVINOExecutionProvider, Session, TensorRTExecutionProvider};


use super::retinaface_resnet;
use super::retinaface_mobilenet;
pub use super::found_face;

const MOBILENET_ONNX: &str = r#"C:\Users\tomokazu\RustroverProjects\ameba_blog_downloader\src\retinaface\mobilenet_retinaface.onnx"#;
const RESNET_ONNX: &str = r#"C:\Users\tomokazu\RustroverProjects\ameba_blog_downloader\src\retinaface\resnet_retinaface.onnx"#;


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
        let execution_providers = [
            OpenVINOExecutionProvider::default().with_device_type("GPU_FP16").build(),
            // TensorRTExecutionProvider::default().build(),
            // CUDAExecutionProvider::default().build(),
            DirectMLExecutionProvider::default().with_device_id(0).build(),
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
                    session: session_builder.with_model_from_file(MOBILENET_ONNX).unwrap(),
                    model: model_kind,
                }
            }
            ModelKind::ResNet => {
                RetinaFaceFaceDetector {
                    session: session_builder.with_model_from_file(RESNET_ONNX).unwrap(),
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

    pub fn image_to_array(&self, image: DynamicImage) -> Array4<f32> {
        match self.model {
            ModelKind::MobileNet => retinaface_mobilenet::transform(image),
            ModelKind::ResNet => retinaface_resnet::transform(image)
        }
    }
}
