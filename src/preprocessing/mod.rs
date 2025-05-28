//! Image preprocessing utilities

use anyhow::Result;
use opencv::{core, imgproc, prelude::*};

/// Image preprocessor for ML inference
#[derive(Clone)]
pub struct Preprocessor {
    target_width: i32,
    target_height: i32,
}

impl Preprocessor {
    /// Create a new preprocessor with target dimensions
    pub fn new(target_width: i32, target_height: i32) -> Self {
        Self {
            target_width,
            target_height,
        }
    }

    /// Letterbox an image to target size preserving aspect ratio
    pub fn letterbox(&self, src: &core::Mat) -> Result<core::Mat> {
        let src_size = src.size()?;
        let (orig_w, orig_h) = (src_size.width, src_size.height);

        // Calculate scale to fit the longer side to target size
        let scale = if orig_w > orig_h {
            self.target_width as f64 / orig_w as f64
        } else {
            self.target_height as f64 / orig_h as f64
        };

        let new_w = (orig_w as f64 * scale).round() as i32;
        let new_h = (orig_h as f64 * scale).round() as i32;

        // Create target-sized Mat with black background
        let mat_type = src.typ();
        let mut dst = core::Mat::new_rows_cols_with_default(
            self.target_height,
            self.target_width,
            mat_type,
            core::Scalar::all(0.0),
        )?;

        // Resize source image
        let mut resized = core::Mat::default();
        imgproc::resize(
            src,
            &mut resized,
            core::Size::new(new_w, new_h),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        // Copy resized image to center of destination
        let x_offset = (self.target_width - new_w) / 2;
        let y_offset = (self.target_height - new_h) / 2;
        let roi_rect = core::Rect::new(x_offset, y_offset, new_w, new_h);
        let mut roi = dst.roi_mut(roi_rect)?;
        resized.copy_to(&mut roi)?;

        Ok(dst)
    }

    /// Convert RGB image to normalized float tensor
    pub fn rgb_to_tensor(&self, mat: &core::Mat) -> Result<Vec<f32>> {
        let data = mat.data_bytes()?;
        let total_pixels = (self.target_width * self.target_height * 3) as usize;

        if data.len() < total_pixels {
            return Err(anyhow::anyhow!(
                "Image data too small: {} < {}",
                data.len(),
                total_pixels
            ));
        }

        // Convert to normalized float values [0.0, 1.0]
        let normalized: Vec<f32> = data[..total_pixels]
            .iter()
            .map(|&pixel| pixel as f32 / 255.0)
            .collect();

        Ok(normalized)
    }

    /// Convert HWC (Height-Width-Channel) to CHW (Channel-Height-Width) format
    pub fn hwc_to_chw(&self, hwc_data: &[f32]) -> Result<Vec<f32>> {
        let height = self.target_height as usize;
        let width = self.target_width as usize;
        let channels = 3;
        let total_size = height * width * channels;

        if hwc_data.len() != total_size {
            return Err(anyhow::anyhow!(
                "Data size mismatch: {} != {}",
                hwc_data.len(),
                total_size
            ));
        }

        let mut chw_data = vec![0.0f32; total_size];

        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    let src_idx = h * width * channels + w * channels + c;
                    let dst_idx = c * height * width + h * width + w;
                    chw_data[dst_idx] = hwc_data[src_idx];
                }
            }
        }

        Ok(chw_data)
    }

    /// Complete preprocessing pipeline: letterbox + normalize + format conversion
    pub fn process(&self, src: &core::Mat) -> Result<Vec<f32>> {
        let letterboxed = self.letterbox(src)?;
        let normalized = self.rgb_to_tensor(&letterboxed)?;
        let chw_data = self.hwc_to_chw(&normalized)?;
        Ok(chw_data)
    }

    /// Get the target dimensions
    pub fn target_size(&self) -> (i32, i32) {
        (self.target_width, self.target_height)
    }
}

/// Default preprocessor for 640x640 YOLO models
impl Default for Preprocessor {
    fn default() -> Self {
        Self::new(640, 640)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_mat(width: i32, height: i32) -> opencv::Result<core::Mat> {
        core::Mat::new_rows_cols_with_default(
            height,
            width,
            core::CV_8UC3,
            core::Scalar::all(128.0),
        )
    }

    #[test]
    fn test_preprocessor_creation() {
        let preprocessor = Preprocessor::new(640, 640);
        assert_eq!(preprocessor.target_size(), (640, 640));
    }

    #[test]
    fn test_default_preprocessor() {
        let preprocessor = Preprocessor::default();
        assert_eq!(preprocessor.target_size(), (640, 640));
    }

    #[test]
    fn test_letterbox_square_image() {
        let preprocessor = Preprocessor::default();
        let src = create_test_mat(640, 640).unwrap();
        let result = preprocessor.letterbox(&src).unwrap();

        assert_eq!(result.size().unwrap().width, 640);
        assert_eq!(result.size().unwrap().height, 640);
    }

    #[test]
    fn test_letterbox_wide_image() {
        let preprocessor = Preprocessor::default();
        let src = create_test_mat(1280, 720).unwrap();
        let result = preprocessor.letterbox(&src).unwrap();

        assert_eq!(result.size().unwrap().width, 640);
        assert_eq!(result.size().unwrap().height, 640);
    }

    #[test]
    fn test_letterbox_tall_image() {
        let preprocessor = Preprocessor::default();
        let src = create_test_mat(480, 800).unwrap();
        let result = preprocessor.letterbox(&src).unwrap();

        assert_eq!(result.size().unwrap().width, 640);
        assert_eq!(result.size().unwrap().height, 640);
    }

    #[test]
    fn test_hwc_to_chw_conversion() {
        let preprocessor = Preprocessor::new(2, 2); // Small test case
        let hwc_data = vec![
            1.0, 2.0, 3.0, // Pixel (0,0): R=1, G=2, B=3
            4.0, 5.0, 6.0, // Pixel (0,1): R=4, G=5, B=6
            7.0, 8.0, 9.0, // Pixel (1,0): R=7, G=8, B=9
            10.0, 11.0, 12.0, // Pixel (1,1): R=10, G=11, B=12
        ];

        let chw_data = preprocessor.hwc_to_chw(&hwc_data).unwrap();

        // CHW format: all R values, then all G values, then all B values
        let expected = vec![
            1.0, 4.0, 7.0, 10.0, // R channel
            2.0, 5.0, 8.0, 11.0, // G channel
            3.0, 6.0, 9.0, 12.0, // B channel
        ];

        assert_eq!(chw_data, expected);
    }

    #[test]
    fn test_complete_preprocessing_pipeline() {
        let preprocessor = Preprocessor::new(4, 4); // Small test case
        let src = create_test_mat(4, 4).unwrap();

        let result = preprocessor.process(&src);
        assert!(result.is_ok());

        let tensor = result.unwrap();
        assert_eq!(tensor.len(), 4 * 4 * 3); // width * height * channels

        // All values should be normalized (0.0 to 1.0)
        assert!(tensor.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }
}
