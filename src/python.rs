//! Python bindings for the Pup video processing library
//!
//! This module provides Python bindings using PyO3 for key functionality
//! of the Pup video processing library.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;

use crate::inference::{InferenceBackend, OrtBackend};

/// Python wrapper for OrtBackend
#[pyclass(name = "OrtBackend")]
pub struct PyOrtBackend {
    backend: OrtBackend,
}

#[pymethods]
impl PyOrtBackend {
    /// Create a new OrtBackend instance
    #[new]
    #[pyo3(signature = (use_coreml=true))]
    pub fn new(use_coreml: bool) -> Self {
        let mut backend = if use_coreml {
            OrtBackend::new()
        } else {
            OrtBackend::with_cpu_only()
        };

        if use_coreml {
            backend.with_coreml(true);
        }

        Self { backend }
    }

    /// Load a model from file
    pub fn load_model(&mut self, path: &str) -> PyResult<()> {
        let model_path = Path::new(path);
        self.backend.load_model(model_path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load model: {}",
                e
            ))
        })?;
        Ok(())
    }

    /// Run inference on input data
    pub fn infer(&self, input: Vec<f32>) -> PyResult<PyObject> {
        let result = self.backend.infer(&input).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Inference failed: {}", e))
        })?;

        Python::with_gil(|py| {
            let dict = PyDict::new_bound(py);

            match result {
                crate::inference::TaskOutput::Detections(detections) => {
                    dict.set_item("type", "detections")?;
                    let detection_list = detections
                        .iter()
                        .map(|det| {
                            let detection_dict = PyDict::new_bound(py);
                            detection_dict.set_item("class_id", det.class_id).unwrap();
                            detection_dict.set_item("confidence", det.score).unwrap();
                            detection_dict
                                .set_item("bbox", vec![det.x1, det.y1, det.x2, det.y2])
                                .unwrap();
                            detection_dict.to_object(py)
                        })
                        .collect::<Vec<_>>();
                    dict.set_item("detections", detection_list)?;
                }
            }

            Ok(dict.to_object(py))
        })
    }

    /// Get the input shape expected by the model
    pub fn get_input_shape(&self) -> Vec<usize> {
        self.backend.get_input_shape().to_vec()
    }
}

/// Benchmark inference performance between CPU and CoreML
#[pyfunction]
#[pyo3(signature = (model_path, runs=20))]
pub fn benchmark_inference(model_path: &str, runs: usize) -> PyResult<PyObject> {
    use std::time::Instant;

    let path = Path::new(model_path);
    if !path.exists() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            format!("Model file not found: {}", model_path),
        ));
    }

    // Benchmark CPU
    let mut cpu_backend = OrtBackend::with_cpu_only();
    cpu_backend.load_model(path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to load model for CPU: {}",
            e
        ))
    })?;

    // Create dummy input data
    let input_data: Vec<f32> = (0..3 * 640 * 640)
        .map(|i| (i as f32) % 256.0 / 255.0)
        .collect();

    let mut cpu_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        cpu_backend.infer(&input_data).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "CPU inference failed: {}",
                e
            ))
        })?;
        cpu_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    // Benchmark CoreML
    let mut coreml_backend = OrtBackend::new();
    coreml_backend.with_coreml(true);
    coreml_backend.load_model(path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to load model for CoreML: {}",
            e
        ))
    })?;

    let mut coreml_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        coreml_backend.infer(&input_data).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "CoreML inference failed: {}",
                e
            ))
        })?;
        coreml_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    // Calculate statistics
    let cpu_avg = cpu_times.iter().sum::<f64>() / cpu_times.len() as f64;
    let coreml_avg = coreml_times.iter().sum::<f64>() / coreml_times.len() as f64;

    let cpu_std = (cpu_times.iter().map(|x| (x - cpu_avg).powi(2)).sum::<f64>()
        / cpu_times.len() as f64)
        .sqrt();
    let coreml_std = (coreml_times
        .iter()
        .map(|x| (x - coreml_avg).powi(2))
        .sum::<f64>()
        / coreml_times.len() as f64)
        .sqrt();

    let speedup = cpu_avg / coreml_avg;

    Python::with_gil(|py| {
        let results = PyDict::new_bound(py);

        let cpu_dict = PyDict::new_bound(py);
        cpu_dict.set_item("avg_ms", cpu_avg)?;
        cpu_dict.set_item("std_ms", cpu_std)?;
        cpu_dict.set_item("times_ms", cpu_times)?;

        let coreml_dict = PyDict::new_bound(py);
        coreml_dict.set_item("avg_ms", coreml_avg)?;
        coreml_dict.set_item("std_ms", coreml_std)?;
        coreml_dict.set_item("times_ms", coreml_times)?;

        results.set_item("cpu", cpu_dict)?;
        results.set_item("coreml", coreml_dict)?;
        results.set_item("speedup", speedup)?;
        results.set_item(
            "improvement_percent",
            ((cpu_avg - coreml_avg) / cpu_avg) * 100.0,
        )?;

        Ok(results.to_object(py))
    })
}
