use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;

use wasm_bindgen_futures::{JsFuture};
use web_sys::{console, Request, RequestInit, Response};

use image::{imageops::FilterType};
use js_sys::{Uint8Array, Float32Array};

use tract_onnx::prelude::*;

mod utils;
use utils::debug_image;

fn bypass_image_url(url: &str) -> String {
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

fn shape_image(array_buffer: &JsValue) -> Result<Float32Array, JsValue> {
    let uint8_array = Uint8Array::new(&array_buffer);
    let image_data = uint8_array.to_vec();

    // Decode image using `image` crate
    let img = image::load_from_memory(&image_data)
        .map_err(|e| JsValue::from_str(&format!("Image decode error: {}", e)))?;

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

    let float_data: Vec<f32> = cropped_img
        .to_rgb8()
        .enumerate_pixels()
        .flat_map(|(_, _, pixel)| {
            [0, 1, 2].map(|c| {
                let pixel_val = pixel[c] as f32 / 255.0;
                (pixel_val - mean[c]) / std[c]
            })
        })
        .collect();
    
    let float32_array = Float32Array::from(float_data.as_slice());
    
    debug_image(&cropped_img)?;
    
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

fn get_imagenet_labels() -> Result<Vec<String>, JsValue> {
    let labels = include_str!("../synset.txt")
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
        .collect();

    Ok(labels)
}

#[wasm_bindgen]
pub async fn startup() -> Result<(), JsValue> {
    // let url = bypass_image("https://gahag.net/img/201509/10s/gahag-0003126694-1.jpg");
    // let url = bypass_image("https://teamhope-f.jp/content/images/cr/82_shiba.jpg");
    let url = bypass_image_url("https://hoken.rakuten.co.jp/uploads/img/column/pet/welsh-corgi-pembroke/img_contents-02.jpeg");

    let array_buffer = fetch_image(&url).await?;
    let shaped_image = shape_image(&array_buffer)?;

    let model = load_model()?;

    let scores = run_model(&model, &shaped_image)?;

    let labels = get_imagenet_labels()?;

    let top5 = scores.iter()
        .take(5)
        .map(|(idx, score)| format!("Class: {}, Score: {:.4}", labels[*idx], score))
        .collect::<Vec<String>>()
        .join("\n");

    console::log_1(&JsValue::from_str(&format!("Top 5 predictions:\n{}", top5)));

    Ok(())
}

