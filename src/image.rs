use image::{DynamicImage, imageops::FilterType};
use js_sys::{Float32Array, Uint8Array};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, Response};

pub async fn fetch_shaped_image(image_url: &str) -> Result<Float32Array, JsValue> {
    let array_buffer = fetch_image(image_url).await?;
    shape_image(&array_buffer)
}

async fn fetch_image(image_url: &str) -> Result<JsValue, JsValue> {
    let bypass_url = bypass_image_url(image_url);

    let options = RequestInit::new();
    options.set_method("GET");

    let request = Request::new_with_str_and_init(&bypass_url, &options)
        .map_err(|e| JsValue::from_str(&format!("Failed to create request: {:?}", e)))?;

    // Worker環境でのfetch処理
    let global = js_sys::global();
    let fetch_promise = if let Ok(worker_scope) = global.clone().dyn_into::<web_sys::DedicatedWorkerGlobalScope>() {
        // Worker環境
        worker_scope.fetch_with_request(&request)
    } else if let Ok(window) = global.dyn_into::<web_sys::Window>() {
        // メインスレッド環境
        window.fetch_with_request(&request)
    } else {
        return Err(JsValue::from_str("Unsupported environment"));
    };

    let response_value = JsFuture::from(fetch_promise).await?;
    let response = response_value.dyn_into::<Response>()?;
    
    if !response.ok() {
        return Err(JsValue::from_str(&format!("HTTP error: {}", response.status())));
    }
    
    let array_buffer = JsFuture::from(response.array_buffer()?).await?;

    Ok(array_buffer)
}

fn shape_image(array_buffer: &JsValue) -> Result<Float32Array, JsValue> {
    let image_data = Uint8Array::new(array_buffer).to_vec();

    // Decode image using `image` crate
    let img = image::load_from_memory(&image_data)
        .map_err(|e| JsValue::from_str(&format!("Image decode error: {}", e)))?;

    let shaped_data = preprocess_image(&img)?;

    Ok(shaped_data)
}

fn bypass_image_url(url: &str) -> String {
    format!("http://localhost:3000/image?url={}", url)
}

fn preprocess_image(img: &DynamicImage) -> Result<Float32Array, JsValue> {
    // ResNetの標準的な前処理
    // 1. 256にリサイズ（短辺が256になるように）
    let height = img.height();
    let width = img.width();
    let scale = 256.0 / height.min(width) as f32;
    let new_height = (height as f32 * scale).round() as u32;
    let new_width = (width as f32 * scale).round() as u32;

    let mut resized_img = img.resize_exact(new_width, new_height, FilterType::Triangle);

    // 2. 中央から224x224を切り出す
    let h_offset = (new_height - 224) / 2;
    let w_offset = (new_width - 224) / 2;
    let cropped_img = resized_img.crop(w_offset, h_offset, 224, 224);

    // 3. RGB値を[0,1]の範囲に正規化し、ImageNetの平均と標準偏差で標準化
    // ImageNetの標準的な平均と標準偏差
    let mean = [0.485, 0.456, 0.406]; // RGB
    let std = [0.229, 0.224, 0.225]; // RGB

    let mut float_data = vec![0.0; 3 * 224 * 224];
    let c_offset = 224 * 224;

    cropped_img
        .to_rgb8()
        .as_raw()
        .chunks_exact(3)
        .enumerate()
        .for_each(|(i, pixel)| {
            // CHWフォーマットで格納
            float_data[i] = (pixel[0] as f32 / 255.0 - mean[0]) / std[0]; // R
            float_data[c_offset + i] = (pixel[1] as f32 / 255.0 - mean[1]) / std[1]; // G
            float_data[2 * c_offset + i] = (pixel[2] as f32 / 255.0 - mean[2]) / std[2];
        });

    let float32_array = Float32Array::from(float_data.as_slice());

    Ok(float32_array)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageBuffer;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    // #[wasm_bindgen_test]
    // #[allow(dead_code)]
    // async fn test_fetch_image() {
    //     let test_url = "https://images.google.com/images/branding/googlelogo/2x/googlelogo_light_color_272x92dp.png";

    //     let result = fetch_image(test_url).await;

    //     assert!(result.is_ok());

    //     let array_buffer = result.unwrap();
    //     let shaped_data = shape_image(&array_buffer);

    //     assert!(shaped_data.is_ok());

    // }

    #[wasm_bindgen_test]
    #[allow(dead_code)]
    fn test_preprocess_image() {
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_fn(300, 300, |x, y| {
            if (x + y) % 2 == 0 {
                image::Rgb([255, 0, 0])
            } else {
                image::Rgb([0, 0, 255])
            }
        }));

        let result = preprocess_image(&img);

        assert!(result.is_ok());

        let float_array = result.unwrap();
        // 224x224x3 = 150528 の要素数を確認
        assert_eq!(float_array.length(), 150528);
    }

    #[wasm_bindgen_test]
    #[allow(dead_code)]
    fn test_bypass_image_url() {
        let original_url = "https://example.com/image.jpg";
        let expected = "http://localhost:3000/image?url=https://example.com/image.jpg";
        assert_eq!(bypass_image_url(original_url), expected);
    }
}
