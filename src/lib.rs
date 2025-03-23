use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;

use wasm_bindgen_futures::{JsFuture};
use web_sys::{console, Request, RequestInit, Response};

use image::{DynamicImage, GenericImageView, imageops::FilterType};
use js_sys::{Uint8Array, Float32Array};

use tract_onnx::prelude::*;

fn bypass_image(url: &str) -> String {
    format!("http://localhost:3000/image?url={}", url)
}

async fn fetch_image(url: &str) -> Result<JsValue, JsValue> {
    let options = RequestInit::new();
    options.set_method("GET");

    let request = Request::new_with_str_and_init(url, &options).unwrap();

    let window = web_sys::window().unwrap();
    let response_value = JsFuture::from(window.fetch_with_request(&request)).await?;

    let response = response_value.dyn_into::<Response>().unwrap();
    let array_buffer = JsFuture::from(response.array_buffer()?).await?;

    Ok(array_buffer)
}

// 224*224にshapeする
// fn image_to_f32_array(img: &DynamicImage) -> Vec<f32> {
//     let (width, height) = img.dimensions();
//     let mut f32_data = Vec::with_capacity((width * height * 3) as usize);

//     for pixel in img.to_rgb8().pixels() {
//         f32_data.push(pixel[0] as f32 / 255.0); // R
//         f32_data.push(pixel[1] as f32 / 255.0); // G
//         f32_data.push(pixel[2] as f32 / 255.0); // B
//     }

//     f32_data
// }

// fn shape_image(array_buffer: &JsValue) -> Result<Float32Array, JsValue> {
//     let uint8_array = Uint8Array::new(&array_buffer);
//     let image_data = uint8_array.to_vec();

//     // Decode image using `image` crate
//     let img = image::load_from_memory(&image_data)
//         .map_err(|e| JsValue::from_str(&format!("Image decode error: {}", e)))?;

//     // // Resize image
//     let resized_img = img.resize(224, 224, FilterType::Triangle);

//     // // Convert to Float32Array
//     let float_data: Vec<f32> = image_to_f32_array(&resized_img);
//     let float32_array = Float32Array::from(float_data.as_slice());
    
//     Ok(float32_array)
// }

fn shape_image(array_buffer: &JsValue) -> Result<Float32Array, JsValue> {
    let uint8_array = Uint8Array::new(&array_buffer);
    let image_data = uint8_array.to_vec();

    // Decode image using `image` crate
    let img = image::load_from_memory(&image_data)
        .map_err(|e| JsValue::from_str(&format!("Image decode error: {}", e)))?;

    // Resize image
    let resized_img = img.resize_exact(224, 224, FilterType::Triangle);

    // Convert to NCHW format (1,3,224,224)
    let mut float_data = Vec::with_capacity(1 * 3 * 224 * 224);
    let rgb_img = resized_img.to_rgb8();
    
    // Channel-first order (NCHW): すべてのRチャネル、次にG、次にB
    // Red channel
    for y in 0..224 {
        for x in 0..224 {
            let pixel = rgb_img.get_pixel(x, y);
            float_data.push(pixel[0] as f32 / 255.0);
        }
    }
    
    // Green channel
    for y in 0..224 {
        for x in 0..224 {
            let pixel = rgb_img.get_pixel(x, y);
            float_data.push(pixel[1] as f32 / 255.0);
        }
    }
    
    // Blue channel
    for y in 0..224 {
        for x in 0..224 {
            let pixel = rgb_img.get_pixel(x, y);
            float_data.push(pixel[2] as f32 / 255.0);
        }
    }

    let float32_array = Float32Array::from(float_data.as_slice());
    
    Ok(float32_array)
}

type TractModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;
fn load_model() -> Result<TractModel, JsValue> {
    let model_bytes = include_bytes!("../resnet18-v1-7.onnx");

    let model = tract_onnx::onnx()
            // .model_for_path("../efficientnet-lite4-11-int8.onnx");
            .model_for_read(&mut std::io::Cursor::new(model_bytes))
            .map_err(|e| JsValue::from_str(&format!("モデルのロードに失敗: {}", e)))?
            .into_optimized()
            .map_err(|e| JsValue::from_str(&format!("モデルの最適化に失敗: {}", e)))?
            .into_runnable()
            .map_err(|e| JsValue::from_str(&format!("実行可能なモデルへの変換に失敗: {}", e)))?;

    Ok(model)
}

// fn run_model(model: &TractModel, input: &Float32Array) -> Result<Vec<(usize, f32)>, JsValue> {
//     let input_tensor = tract_onnx::prelude::Tensor::from_shape(
//         &[1, 3, 224, 224],
//         input_data
//     ).map_err(|e| JsValue::from_str(&format!("テンソル作成エラー: {}", e)))?;
//     let input_tensor = input_fact.into_tensor().into_owned();

//     let output = model.run(tvec![input_tensor])?;

//     let mut scores: Vec<(usize, f32)> = output
//         .iter()
//         .enumerate()
//         .map(|(i, &score)| (i, score))
//         .collect();
    
//     scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

//     Ok(scores)
// }

fn run_model(model: &TractModel, input: &Float32Array) -> Result<Vec<(usize, f32)>, JsValue> {
    // 入力データを取得
    let input_data = input.to_vec();
    let input_tensor = tract_onnx::prelude::Tensor::from_shape(
        &[1, 3, 224, 224],
        &input_data  // ここに & を追加
    ).map_err(|e| JsValue::from_str(&format!("テンソル作成エラー: {}", e)))?;

    // テンソルを TValue に変換して実行
    let output = model.run(tvec![input_tensor.into()])  // .into() を追加
        .map_err(|e| JsValue::from_str(&format!("モデル実行エラー: {}", e)))?;

    // 出力テンソルから正しく値を取得
    // 最初の出力テンソルを取得
    let tensor = output[0].clone();
    let values = tensor.as_slice::<f32>()
        .map_err(|e| JsValue::from_str(&format!("テンソル変換エラー: {}", e)))?;

    // スコアをインデックスと共に保存
    let mut scores: Vec<(usize, f32)> = values.iter()
        .enumerate()
        .map(|(i, &score)| (i, score))
        .collect();
    
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    Ok(scores)
}

fn get_imagenet_labels() -> Vec<String> {
    include_str!("../synset.txt")
        .lines()
        .map(|line| {
            // WordNet IDを除去してラベル部分のみを取得
            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            if parts.len() > 1 {
                parts[1].to_string()
            } else {
                line.to_string()
            }
        })
        .collect()
}

#[wasm_bindgen]
pub async fn startup() -> Result<(), JsValue> {
    let url = bypass_image("https://gahag.net/img/201509/10s/gahag-0003126694-1.jpg");

    let array_buffer = fetch_image(&url).await?;
    let shaped_image = shape_image(&array_buffer)?;

    let model = load_model()?;

    let scores = run_model(&model, &shaped_image)?;

    let labels = get_imagenet_labels();

    let top5 = scores.iter()
        .take(5)
        .map(|(idx, score)| format!("クラス: {}, スコア: {:.4}", labels[*idx], score))
        .collect::<Vec<String>>()
        .join("\n");

        console::log_1(&JsValue::from_str(&format!("Top 5 predictions:\n{}", top5)));

    Ok(())
}

