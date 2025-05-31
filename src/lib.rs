use wasm_bindgen::JsValue;
use wasm_bindgen::prelude::*;
use web_sys::{MessageEvent, DedicatedWorkerGlobalScope};

mod image;
mod model;

// Memo: debugを利用してなくても、テストを動作させるためにはモジュールを宣言する必要がある
// see: https://rustwasm.github.io/wasm-bindgen/wasm-bindgen-test/usage.html#:~:text=One%20other%20difference%20is%20that%20the%20tests%20must%20be%20in%20the%20root%20of%20the%20crate%2C%20or%20within%20a%20pub%20mod.%20Putting%20them%20inside%20a%20private%20module%20will%20not%20work.
mod debug;

// メインスレッド用：Workerを起動する
#[wasm_bindgen]
pub fn startup() {
    console_error_panic_hook::set_once();
    web_sys::console::log_1(&"Main thread started".into());
}

// Worker用：実際の推論処理
#[wasm_bindgen]
pub struct ImageAnalyzer;

#[wasm_bindgen]
impl ImageAnalyzer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ImageAnalyzer {
        console_error_panic_hook::set_once();
        ImageAnalyzer
    }

    #[wasm_bindgen]
    pub async fn analyze_image(&self, url: &str) -> Result<JsValue, JsValue> {
        let shaped_image = image::fetch_shaped_image(url).await?;
        let result = model::infer_top5(&shaped_image)?;
        Ok(convert_to_js_value(&result))
    }
}

// Worker内でのメッセージハンドリング用
#[wasm_bindgen]
pub fn setup_worker_message_handler() {
    let callback = Closure::wrap(Box::new(move |event: MessageEvent| {
        let data = event.data();
        
        if let Some(url) = data.as_string() {
            wasm_bindgen_futures::spawn_local(async move {
                let analyzer = ImageAnalyzer::new();
                match analyzer.analyze_image(&url).await {
                    Ok(result) => {
                        let scope = js_sys::global().unchecked_into::<DedicatedWorkerGlobalScope>();
                        if let Err(_) = scope.post_message(&result) {
                            web_sys::console::error_1(&"Failed to post message".into());
                        }
                    }
                    Err(error) => {
                        let scope = js_sys::global().unchecked_into::<DedicatedWorkerGlobalScope>();
                        if let Err(_) = scope.post_message(&error) {
                            web_sys::console::error_1(&"Failed to post error message".into());
                        }
                    }
                }
            });
        }
    }) as Box<dyn FnMut(MessageEvent)>);
    
    let scope = js_sys::global().unchecked_into::<DedicatedWorkerGlobalScope>();
    scope.set_onmessage(Some(callback.as_ref().unchecked_ref()));
    callback.forget();
}

fn convert_to_js_value(result: &model::InferResultWithLabels) -> JsValue {
    let predictions = result
        .iter()
        .enumerate()
        .map(|(index, (label, score))| {
            serde_json::json!({
                "label": label,
                "score": score,
                "rank": index + 1
            })
        })
        .collect::<Vec<_>>();

    let output = serde_json::to_string(&predictions).unwrap();
    JsValue::from_str(&output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    // #[wasm_bindgen_test]
    // #[allow(dead_code)]
    // async fn test_analyze_image() {
    //     let url = "https://wantimes.leoandlea.com/static/87016e3dfe93fa9ea21268225ffe8cc9/b8f0a/a2f618f8-802e-43a6-b902-39b03e3ffc6d_breedswelsh-corgi-pembroke.jpg";
    //     let result = analyze_image(url).await;
    //     assert!(result.is_ok());
    // }

    #[wasm_bindgen_test]
    #[allow(dead_code)]
    fn test_convert_to_js_value() {
        let result: model::InferResultWithLabels = vec![
            ("猫".to_string(), 0.95),
            ("犬".to_string(), 0.80),
            ("鳥".to_string(), 0.60),
        ];

        let js_value = convert_to_js_value(&result);
        let result_str = js_value.as_string().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result_str).unwrap();

        assert!(parsed.is_array());
        let array = parsed.as_array().unwrap();

        assert_eq!(array.len(), 3);

        assert_eq!(array[0]["label"], "猫");
        // 浮動小数点は厳密な等価比較ができないため、許容誤差内(1e-6)での比較を実施
        assert!((array[0]["score"].as_f64().unwrap() - 0.95).abs() < 1e-6);
        assert_eq!(array[0]["rank"], 1);
    }
}
