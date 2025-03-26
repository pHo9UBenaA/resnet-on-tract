use wasm_bindgen::JsValue;
use wasm_bindgen::prelude::*;

mod image;
mod model;

// Memo: debugを利用してなくても、テストを動作させるためにはモジュールを宣言する必要がある
// see: https://rustwasm.github.io/wasm-bindgen/wasm-bindgen-test/usage.html#:~:text=One%20other%20difference%20is%20that%20the%20tests%20must%20be%20in%20the%20root%20of%20the%20crate%2C%20or%20within%20a%20pub%20mod.%20Putting%20them%20inside%20a%20private%20module%20will%20not%20work.
mod debug;

#[wasm_bindgen]
pub async fn analyze_image(url: &str) -> Result<JsValue, JsValue> {
    let shaped_image = image::fetch_shaped_image(url).await?;

    let result = model::infer_top5(&shaped_image)?;

    Ok(convert_to_js_value(&result))
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

    #[wasm_bindgen_test]
    fn test_convert_to_js_value() {
        let result: model::InferResultWithLabels = vec![
            ("猫".to_string(), 0.95),
            ("犬".to_string(), 0.80),
            ("鳥".to_string(), 0.60)
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
