use js_sys::Float32Array;
use tract_onnx::prelude::*;
use wasm_bindgen::prelude::*;
type TractModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

type InferResult = Vec<(usize, f32)>;
type InferResultWithLabels = Vec<(String, f32)>;

pub fn infer_top5(shaped_image: &Float32Array) -> Result<InferResultWithLabels, JsValue> {
    let model = model_load()?;

    let scores = model_run(&model, shaped_image)?;

    let labels = load_labels()?;

    let top5_labels = scores
        .iter()
        .take(5)
        .map(|(idx, score)| (labels[*idx].clone(), *score))
        .collect::<InferResultWithLabels>();

    Ok(top5_labels)
}

fn model_load() -> Result<TractModel, JsValue> {
    let model_bytes = include_bytes!("../resnet18-v1-7.onnx");

    let model = tract_onnx::onnx()
        .model_for_read(&mut std::io::Cursor::new(model_bytes))
        .map_err(|e| JsValue::from_str(&format!("モデルのロードに失敗: {}", e)))?
        .into_optimized()
        .map_err(|e| JsValue::from_str(&format!("モデルの最適化に失敗: {}", e)))?
        .into_runnable()
        .map_err(|e| JsValue::from_str(&format!("実行可能なモデルへの変換に失敗: {}", e)))?;

    Ok(model)
}

fn model_run(model: &TractModel, input: &Float32Array) -> Result<InferResult, JsValue> {
    // 入力データを取得
    let input_data = input.to_vec();
    let input_tensor = tract_onnx::prelude::Tensor::from_shape(&[1, 3, 224, 224], &input_data)
        .map_err(|e| JsValue::from_str(&format!("テンソル作成エラー: {}", e)))?;

    // テンソルを TValue に変換して実行
    let output = model
        .run(tvec![input_tensor.into()])
        .map_err(|e| JsValue::from_str(&format!("モデル実行エラー: {}", e)))?;

    // 出力テンソルから正しく値を取得
    // 最初の出力テンソルを取得
    let tensor = output[0].clone();
    let values = tensor
        .as_slice::<f32>()
        .map_err(|e| JsValue::from_str(&format!("テンソル変換エラー: {}", e)))?;

    // スコアをインデックスと共に保存
    let mut scores: InferResult = values
        .iter()
        .enumerate()
        .map(|(i, &score)| (i, score))
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    Ok(scores)
}

fn load_labels() -> Result<Vec<String>, JsValue> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    fn create_test_input() -> Float32Array {
        // 224x224の3チャンネル画像を想定した入力データ (1x3x224x224 = 150528要素)
        let buffer = Float32Array::new_with_length(3 * 224 * 224);

        // テスト用に全ての値を0.1に設定
        for i in 0..buffer.length() {
            buffer.set_index(i, 0.1);
        }

        buffer
    }

    #[wasm_bindgen_test]
    fn test_model_load() {
        let result = model_load();
        assert!(result.is_ok())
    }

    #[wasm_bindgen_test]
    fn test_load_labels() {
        let result = load_labels();
        assert!(result.is_ok());

        let labels = result.unwrap();
        assert!(!labels.is_empty());
        assert_eq!(labels.len(), 1000);
    }

    #[wasm_bindgen_test]
    fn test_model_run() {
        let model = model_load().unwrap();
        let input = create_test_input();

        let result = model_run(&model, &input);
        assert!(result.is_ok());

        let scores = result.unwrap();
        assert!(!scores.is_empty());
        assert_eq!(scores.len(), 1000);

        // スコアがソートされていることを確認
        // 全て確認する必要はないため、サンプルとして一番目と二番目、二番目と三番目のスコアを比較
        assert!(scores[0].1 >= scores[1].1);
        assert!(scores[1].1 >= scores[2].1);
    }

    #[wasm_bindgen_test]
    fn test_infer_top5() {
        let input = create_test_input();

        let result = infer_top5(&input);
        assert!(result.is_ok());

        let top5 = result.unwrap();
        assert_eq!(top5.len(), 5);

        // すべての結果にラベルとスコアがあることを確認
        // 5個しか要素がないので全ての要素を確認している
        for (label, score) in &top5 {
            assert!(!label.is_empty());
            assert!(score.is_finite());
        }

        // スコアがソートされていることを確認
        for i in 0..top5.len() - 1 {
            assert!(
                top5[i].1 >= top5[i + 1].1,
                "トップ5のスコアが降順ソートされていない"
            );
        }
    }

    #[wasm_bindgen_test]
    fn test_error_handling() {
        // 明らかに小さすぎる
        let small_buffer = Float32Array::new_with_length(10);

        let result = infer_top5(&small_buffer);
        assert!(result.is_err());
    }
}
