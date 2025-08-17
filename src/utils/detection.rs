//! Detection data structures and utilities

use std::fmt;

/// A detected object with bounding box and classification
#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    /// Left X coordinate
    pub x1: f32,
    /// Top Y coordinate  
    pub y1: f32,
    /// Right X coordinate
    pub x2: f32,
    /// Bottom Y coordinate
    pub y2: f32,
    /// Confidence score (0.0 to 1.0)
    pub score: f32,
    /// Class ID
    pub class_id: i32,
}

impl Detection {
    /// Create a new detection
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32, score: f32, class_id: i32) -> Self {
        Self {
            x1,
            y1,
            x2,
            y2,
            score,
            class_id,
        }
    }

    /// Calculate the area of the bounding box
    pub fn area(&self) -> f32 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }

    /// Get the center point of the bounding box
    pub fn center(&self) -> (f32, f32) {
        ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
    }

    /// Get the width of the bounding box
    pub fn width(&self) -> f32 {
        self.x2 - self.x1
    }

    /// Get the height of the bounding box
    pub fn height(&self) -> f32 {
        self.y2 - self.y1
    }

    /// Calculate Intersection over Union (IoU) with another detection
    pub fn iou(&self, other: &Detection) -> f32 {
        let x1 = self.x1.max(other.x1);
        let y1 = self.y1.max(other.y1);
        let x2 = self.x2.min(other.x2);
        let y2 = self.y2.min(other.y2);

        if x2 <= x1 || y2 <= y1 {
            return 0.0;
        }

        let intersection = (x2 - x1) * (y2 - y1);
        let union = self.area() + other.area() - intersection;

        intersection / union
    }

    /// Check if this detection overlaps with another
    pub fn overlaps_with(&self, other: &Detection, threshold: f32) -> bool {
        self.iou(other) > threshold
    }
}

impl fmt::Display for Detection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Detection(class={}, score={:.2}, bbox=({:.1}, {:.1}, {:.1}, {:.1}))",
            self.class_id,
            self.score,
            self.x1,
            self.y1,
            self.width(),
            self.height()
        )
    }
}

/// Detection processing errors
#[derive(Debug, Clone)]
pub enum DetectionError {
    /// Invalid bounding box coordinates
    InvalidBoundingBox(String),
    /// Invalid confidence score
    InvalidScore(f32),
    /// Invalid class ID
    InvalidClassId(i32),
}

impl fmt::Display for DetectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DetectionError::InvalidBoundingBox(msg) => write!(f, "Invalid bounding box: {}", msg),
            DetectionError::InvalidScore(score) => write!(f, "Invalid score: {}", score),
            DetectionError::InvalidClassId(id) => write!(f, "Invalid class ID: {}", id),
        }
    }
}

impl std::error::Error for DetectionError {}

/// Apply Non-Maximum Suppression to a list of detections
pub fn apply_nms(mut detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    // Sort by confidence score (highest first)
    detections.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = Vec::new();
    let mut suppress = vec![false; detections.len()];

    for i in 0..detections.len() {
        if suppress[i] {
            continue;
        }

        keep.push(detections[i].clone());

        // Suppress overlapping detections
        for j in (i + 1)..detections.len() {
            if suppress[j] {
                continue;
            }

            if detections[i].overlaps_with(&detections[j], iou_threshold) {
                suppress[j] = true;
            }
        }
    }

    keep
}

/// Filter detections by confidence threshold
pub fn filter_by_confidence(detections: Vec<Detection>, threshold: f32) -> Vec<Detection> {
    detections
        .into_iter()
        .filter(|d| d.score >= threshold)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detection_creation() {
        let det = Detection::new(10.0, 20.0, 30.0, 40.0, 0.8, 1);

        assert_eq!(det.x1, 10.0);
        assert_eq!(det.y1, 20.0);
        assert_eq!(det.x2, 30.0);
        assert_eq!(det.y2, 40.0);
        assert_eq!(det.score, 0.8);
        assert_eq!(det.class_id, 1);
    }

    #[test]
    fn test_detection_area() {
        let det = Detection::new(0.0, 0.0, 10.0, 20.0, 0.9, 0);
        assert_eq!(det.area(), 200.0);
    }

    #[test]
    fn test_detection_center() {
        let det = Detection::new(0.0, 0.0, 10.0, 20.0, 0.9, 0);
        assert_eq!(det.center(), (5.0, 10.0));
    }

    #[test]
    fn test_detection_dimensions() {
        let det = Detection::new(5.0, 10.0, 15.0, 30.0, 0.7, 2);
        assert_eq!(det.width(), 10.0);
        assert_eq!(det.height(), 20.0);
    }

    #[test]
    fn test_iou_calculation() {
        let det1 = Detection::new(0.0, 0.0, 10.0, 10.0, 0.9, 0);
        let det2 = Detection::new(5.0, 5.0, 15.0, 15.0, 0.8, 0);

        let iou = det1.iou(&det2);
        assert!((iou - 0.142857).abs() < 0.001); // 25 / (100 + 100 - 25) = 25/175 H 0.143
    }

    #[test]
    fn test_nms() {
        let detections = vec![
            Detection::new(0.0, 0.0, 10.0, 10.0, 0.9, 0),
            Detection::new(2.0, 2.0, 12.0, 12.0, 0.8, 0), // Overlaps with first
            Detection::new(20.0, 20.0, 30.0, 30.0, 0.7, 1), // Different location
        ];

        let result = apply_nms(detections, 0.4); // Lower threshold to trigger suppression
        assert_eq!(result.len(), 2); // Should keep highest confidence and non-overlapping
        assert_eq!(result[0].score, 0.9); // Highest confidence first
        assert_eq!(result[1].score, 0.7); // Non-overlapping detection
    }

    #[test]
    fn test_confidence_filtering() {
        let detections = vec![
            Detection::new(0.0, 0.0, 10.0, 10.0, 0.9, 0),
            Detection::new(10.0, 10.0, 20.0, 20.0, 0.3, 1),
            Detection::new(20.0, 20.0, 30.0, 30.0, 0.7, 2),
        ];

        let result = filter_by_confidence(detections, 0.5);
        assert_eq!(result.len(), 2);
        assert!(result.iter().all(|d| d.score >= 0.5));
    }
}
