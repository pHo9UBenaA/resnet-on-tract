use wasm_bindgen::JsValue;
use wasm_bindgen::prelude::*;
use web_sys::console;

mod image;
mod model;

// Memo: debugを利用してなくても、テストを動作させるためにはモジュールを宣言する必要がある
// see: https://rustwasm.github.io/wasm-bindgen/wasm-bindgen-test/usage.html#:~:text=One%20other%20difference%20is%20that%20the%20tests%20must%20be%20in%20the%20root%20of%20the%20crate%2C%20or%20within%20a%20pub%20mod.%20Putting%20them%20inside%20a%20private%20module%20will%20not%20work.
mod debug;

#[wasm_bindgen]
pub async fn startup() -> Result<(), JsValue> {
    // let url = bypass_image("https://gahag.net/img/201509/10s/gahag-0003126694-1.jpg");
    // let url = bypass_image("https://teamhope-f.jp/content/images/cr/82_shiba.jpg");
    let url = "https://hoken.rakuten.co.jp/uploads/img/column/pet/welsh-corgi-pembroke/img_contents-02.jpeg";

    let shaped_image = image::fetch_shaped_image(url).await?;

    let top5 = model::infer_top5(&shaped_image)?;

    let top5_str = top5
        .iter()
        .map(|(label, score)| format!("{}: {}", label, score))
        .collect::<Vec<String>>()
        .join("\n");

    console::log_1(&JsValue::from_str(&format!(
        "Top 5 predictions:\n{}",
        top5_str
    )));

    Ok(())
}
