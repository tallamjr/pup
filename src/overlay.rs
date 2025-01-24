/// A simple rectangle-drawing helper to mutate an RGB pixel buffer in-place.
/// `pixels` is row-major `[RGB, RGB, ...]` of length = width * height * 3.
pub fn draw_rectangle(
    pixels: &mut [u8],
    width: usize,
    height: usize,
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    rgb: [u8; 3],
) {
    // Clamp to image boundaries
    let x1 = x1.clamp(0, (width - 1) as i32) as usize;
    let y1 = y1.clamp(0, (height - 1) as i32) as usize;
    let x2 = x2.clamp(0, (width - 1) as i32) as usize;
    let y2 = y2.clamp(0, (height - 1) as i32) as usize;

    // We'll draw a 2-pixel–thick border
    let thickness = 2;

    // Top & bottom
    for x in x1..=x2 {
        for t in 0..thickness {
            let y_top = y1 + t;
            let y_bot = y2.saturating_sub(t);
            set_rgb(pixels, width, x, y_top, rgb);
            set_rgb(pixels, width, x, y_bot, rgb);
        }
    }

    // Left & right
    for y in y1..=y2 {
        for t in 0..thickness {
            let x_left = x1 + t;
            let x_right = x2.saturating_sub(t);
            set_rgb(pixels, width, x_left, y, rgb);
            set_rgb(pixels, width, x_right, y, rgb);
        }
    }
}

/// Helper to set an (x, y) pixel to `[R, G, B]`.
fn set_rgb(pixels: &mut [u8], width: usize, x: usize, y: usize, rgb: [u8; 3]) {
    let idx = (y * width + x) * 3;
    if idx + 2 < pixels.len() {
        pixels[idx] = rgb[0];
        pixels[idx + 1] = rgb[1];
        pixels[idx + 2] = rgb[2];
    }
}
