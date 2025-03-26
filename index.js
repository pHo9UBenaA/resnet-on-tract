const { analyze_image } = wasm_bindgen;

async function init() {
	await wasm_bindgen();
	form_init();
}

function form_init() {
	const form = document.getElementById("url-form");
	const resultContainer = document.getElementById("result-container");

	form.addEventListener("submit", async (event) => {
		resultContainer.textContent = "分析中...";

		event.preventDefault();
		const url = document.getElementById("image-url").value;

		try {
			// string: [{ label: string, score: number, rank: number }, ...]
			const top5 = await analyze_image(url);

			resultContainer.innerHTML = JSON.parse(top5)
				.map((item) => {
					return `<li>${item.label}: ${item.score}</li>`;
				})
				.join("");
		} catch (error) {
			resultContainer.textContent = "分析に失敗しました";
		}
	});
}

init();
form_init();
