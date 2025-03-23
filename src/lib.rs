use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{console, Request, RequestInit, RequestMode, Response};

#[wasm_bindgen]
pub async fn startup() -> Result<(), JsValue> {
    let options = RequestInit::new();
    options.set_method("GET");
    options.set_mode(RequestMode::Cors);

    let url = format!("https://mdn.github.io/learning-area/javascript/oojs/json/superheroes.json");

    let request = Request::new_with_str_and_init(&url, &options).unwrap();

    let window = web_sys::window().unwrap();
    let response_value = JsFuture::from(window.fetch_with_request(&request)).await?;

    let response = response_value.dyn_into::<Response>().unwrap();

    let json = JsFuture::from(response.json()?).await?;

    console::log_1(&json.into());

    Ok(())
}
