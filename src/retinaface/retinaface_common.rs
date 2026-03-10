use anyhow::Error;
use ndarray::{Array, Array4, IxDyn};
use once_cell::sync::Lazy;
use ort::ep::{CPUExecutionProvider, OpenVINOExecutionProvider, TensorRTExecutionProvider};
use ort::session::builder::GraphOptimizationLevel;
#[allow(unused_imports)]
use ort::session::Session;
use std::path::PathBuf;
use std::sync::Once;

pub use super::found_face;
use super::retinaface_mobilenet;
use super::retinaface_resnet;
use crate::project_dir;
use crate::retinaface::found_face::FoundFace;

const MOBILENET_ONNX: Lazy<PathBuf> = Lazy::new(|| project_dir().join("src").join("retinaface").join("mobilenet_retinaface.onnx"));
const RESNET_ONNX: Lazy<PathBuf> = Lazy::new(|| project_dir().join("src").join("retinaface").join("resnet_retinaface.onnx"));

static ORT_INIT: Once = Once::new();

/// ONNX Runtime を初期化する。複数回呼んでも安全（初回のみ実行される）。
pub fn init_ort() {
    ORT_INIT.call_once(|| {
        let lib_path = project_dir().join("onnxruntime-lib").join("lib");
        #[cfg(target_os = "windows")]
        let lib_path = lib_path.join("onnxruntime.dll");
        #[cfg(target_os = "linux")]
        let lib_path = lib_path.join("libonnxruntime.so");
        #[cfg(target_os = "macos")]
        let lib_path = lib_path.join("libonnxruntime.dylib");
        ort::init_from(lib_path)
            .unwrap()
            .commit();
    });
}


pub enum ModelKind {
    MobileNet,
    ResNet,
}

pub struct RetinaFaceFaceDetector {
    pub session: Session,
    pub model: ModelKind,
}


impl RetinaFaceFaceDetector {
    pub fn new(model_kind: ModelKind, model_path: PathBuf) -> RetinaFaceFaceDetector {
        let execution_providers = [
            // OpenVINOExecutionProvider::default().build(),
            // OneDNNExecutionProvider::default().build(),
            TensorRTExecutionProvider::default().build(),
            // CUDAExecutionProvider::default().build(),
            OpenVINOExecutionProvider::default().with_device_type("GPU").with_precision("FP16").build().error_on_failure(),
            // DirectMLExecutionProvider::default().build(),
            CPUExecutionProvider::default().build().error_on_failure(),
        ];
        // for execution_provider in &execution_providers {
        //     if execution_provider.is_available().unwrap() {
        //         println!("Selected Execution Provider: {}", execution_provider.as_str());
        //         break;
        //     }
        // }
        let mut session_builder = Session::builder().unwrap()
            .with_execution_providers(execution_providers).unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
            .with_intra_threads(1).unwrap();

        match model_kind {
            ModelKind::MobileNet => {
                RetinaFaceFaceDetector {
                    session: session_builder.commit_from_file(model_path).unwrap(),
                    model: model_kind,
                }
            }
            ModelKind::ResNet => {
                RetinaFaceFaceDetector {
                    session: session_builder.commit_from_file(model_path).unwrap(),
                    model: model_kind,
                }
            }
        }
    }
    pub fn infer(&mut self, image: Array4<f32>) -> anyhow::Result<(Array<f32, IxDyn>, Array<f32, IxDyn>, Array<f32, IxDyn>, Vec<usize>)> {
        match self.model {
            ModelKind::MobileNet => { retinaface_mobilenet::infer(&mut self.session, image) }
            ModelKind::ResNet => { retinaface_resnet::infer(&mut self.session, image) }
        }
    }

    pub fn post_process(&self, confidence: Array<f32, IxDyn>, loc: Array<f32, IxDyn>, landmark: Array<f32, IxDyn>, input_shape: Vec<usize>) -> anyhow::Result<Vec<Vec<FoundFace>>> {
        match self.model {
            ModelKind::MobileNet => { retinaface_mobilenet::post_process(confidence, loc, landmark, input_shape) }
            ModelKind::ResNet => { retinaface_resnet::post_process(confidence, loc, landmark, input_shape) }
        }
    }

    pub fn find_face(&mut self, image: Array4<f32>) -> Vec<Vec<FoundFace>> {
        let (confidence, loc, landmark, input_shape) = self.infer(image).unwrap();
        self.post_process(confidence, loc, landmark, input_shape).unwrap()
    }

    pub fn image_to_array(&self, image: Vec<u8>) -> Result<Array4<f32>, Error> {
        match self.model {
            ModelKind::MobileNet => retinaface_mobilenet::transform(image),
            ModelKind::ResNet => retinaface_resnet::transform(image)
        }
    }
}
