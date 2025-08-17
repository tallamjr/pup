//! Test to verify YOLO post-processing is working correctly
//! This test will expose the issues with the current implementation

use gstpup::inference::{ModelPostProcessor, YoloPostProcessor};

#[test]
fn test_yolo_output_format_assumptions() {
    let processor = YoloPostProcessor::coco_default();

    // Create a mock YOLOv8 output with REALISTIC values
    // YOLOv8 typically outputs [1, 84, 8400] reshaped to [84, 8400]
    // where 84 = 4 (bbox) + 80 (classes)

    // Let's test with just one detection to see what happens
    let mut output = vec![0.0f32; 84];

    // YOLOv8 format: [cx, cy, w, h, class0_conf, class1_conf, ...]
    // Setting a detection at center (320, 320) with size (100, 100)
    output[0] = 320.0; // center_x (should be in pixel coordinates)
    output[1] = 320.0; // center_y
    output[2] = 100.0; // width
    output[3] = 100.0; // height

    // High confidence for class 0 (person)
    output[4] = 0.95; // class 0 confidence

    let input_shape = vec![1, 3, 640, 640];
    let detections = processor.process_raw_output(&output, &input_shape).unwrap();

    println!("=== YOLO POST-PROCESSING TEST RESULTS ===");
    println!("Input mock detection: center=(320,320), size=(100,100), class_0_conf=0.95");
    println!("Number of detections found: {}", detections.len());

    if !detections.is_empty() {
        let det = &detections[0];
        println!(
            "Output detection: x1={}, y1={}, x2={}, y2={}, score={}, class={}",
            det.x1, det.y1, det.x2, det.y2, det.score, det.class_id
        );

        // Calculate the actual center and size from the output
        let output_center_x = (det.x1 + det.x2) / 2.0;
        let output_center_y = (det.y1 + det.y2) / 2.0;
        let output_width = det.x2 - det.x1;
        let output_height = det.y2 - det.y1;

        println!("Derived center: ({}, {})", output_center_x, output_center_y);
        println!("Derived size: ({}, {})", output_width, output_height);

        // Check if the coordinates make sense
        if det.x1 < 0.0 || det.y1 < 0.0 || det.x2 > 640.0 || det.y2 > 640.0 {
            println!("🚨 COORDINATES OUT OF BOUNDS! This proves the post-processing is wrong!");
        }

        if output_center_x != 320.0 || output_center_y != 320.0 {
            println!("🚨 CENTER COORDINATES TRANSFORMED INCORRECTLY!");
            println!(
                "Expected center: (320, 320), Got: ({}, {})",
                output_center_x, output_center_y
            );
        }
    }

    // This test will likely show that the current post-processing is scaling
    // coordinates incorrectly, leading to random bounding box positions
}

#[test]
fn test_multiple_detections_coordinate_explosion() {
    let processor = YoloPostProcessor::coco_default();

    // Test with multiple "detections" to see if we get the random explosion effect
    let detections_count = 10;
    let mut output = vec![0.0f32; 84 * detections_count];

    // Fill with various coordinate values to simulate actual model output
    for i in 0..detections_count {
        let base_idx = i * 84;

        // Simulate coordinates that might come from a real model
        output[base_idx] = 0.3 + (i as f32 * 0.1); // normalized-looking coords
        output[base_idx + 1] = 0.4 + (i as f32 * 0.1);
        output[base_idx + 2] = 0.1; // width
        output[base_idx + 3] = 0.15; // height

        // Set confidence for random classes
        output[base_idx + 4 + (i % 80)] = 0.6 + (i as f32 * 0.05);
    }

    let input_shape = vec![1, 3, 640, 640];
    let detections = processor.process_raw_output(&output, &input_shape).unwrap();

    println!("=== MULTIPLE DETECTIONS TEST ===");
    println!("Number of detections: {}", detections.len());

    let mut out_of_bounds_count = 0;
    let mut valid_count = 0;

    for (i, det) in detections.iter().enumerate() {
        if det.x1 < 0.0 || det.y1 < 0.0 || det.x2 > 640.0 || det.y2 > 640.0 {
            out_of_bounds_count += 1;
            println!(
                "Detection {}: OUT OF BOUNDS - ({}, {}, {}, {})",
                i, det.x1, det.y1, det.x2, det.y2
            );
        } else {
            valid_count += 1;
        }
    }

    println!(
        "Valid detections: {}, Out of bounds: {}",
        valid_count, out_of_bounds_count
    );

    if out_of_bounds_count > 0 {
        println!(
            "🚨 FOUND {} OUT-OF-BOUNDS DETECTIONS - THIS CONFIRMS THE BUG!",
            out_of_bounds_count
        );
    }
}

#[test]
fn test_confidence_threshold_behavior() {
    let processor = YoloPostProcessor::coco_default();

    // Test with very low confidence values to see if they're being processed incorrectly
    let mut output = vec![0.0f32; 84];

    output[0] = 0.5; // center_x
    output[1] = 0.5; // center_y
    output[2] = 0.2; // width
    output[3] = 0.2; // height
    output[4] = 0.01; // Very low confidence

    let input_shape = vec![1, 3, 640, 640];
    let detections = processor.process_raw_output(&output, &input_shape).unwrap();

    println!("=== CONFIDENCE THRESHOLD TEST ===");
    println!("Input confidence: 0.01 (very low)");
    println!("Detections found: {}", detections.len());

    if !detections.is_empty() {
        println!(
            "🚨 LOW CONFIDENCE DETECTION PASSED THROUGH! Score: {}",
            detections[0].score
        );
        println!("This suggests confidence filtering isn't working as expected");
    }
}
