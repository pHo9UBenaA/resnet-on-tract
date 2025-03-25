use wasm_bindgen::JsValue;
use web_sys::console;
use image::DynamicImage;
use base64::Engine;

#[allow(dead_code)] // テストを動作させることを目的に常にモジュールを宣言するため
pub fn dynamic_image_to_base64(cropped_img: &DynamicImage) -> Result<(), JsValue> {
    let rgb_img = cropped_img.to_rgb8();
    
    // 画像をPNGにエンコードしてbase64に変換
    let mut png_data = Vec::new();
    {
        // 新しいバージョンのimageクレートでは、write_to_pngメソッドを使用する
        rgb_img.write_to(&mut std::io::Cursor::new(&mut png_data), image::ImageFormat::Png)
            .map_err(|e| JsValue::from_str(&format!("PNG encode error: {}", e)))?;
    }
    
    let base64_image = base64::engine::general_purpose::STANDARD.encode(&png_data);
    let url = format!("data:image/png;base64,{}", base64_image);
    console::log_1(&JsValue::from_str(&format!("画像URL: {}", url)));

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer};
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_dynamic_image_to_base64() {
        // テスト用の小さな画像を作成
        let img_buffer: ImageBuffer<image::Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(10, 10, |x, y| {
            // 簡単なパターンを生成
            if (x + y) % 2 == 0 {
                image::Rgb([255, 0, 0])  // 赤
            } else {
                image::Rgb([0, 0, 255])  // 青
            }
        });
        
        let dynamic_img = DynamicImage::ImageRgb8(img_buffer);
        
        let result = dynamic_image_to_base64(&dynamic_img);
        
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_dynamic_image_to_base64_error() {
        let result = dynamic_image_to_base64(&DynamicImage::ImageRgb8(ImageBuffer::new(0, 0)));
        assert!(result.is_err());
    }
}
