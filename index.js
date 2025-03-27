const { analyze_image: analyzeImageOnWasm } = wasm_bindgen;

async function init() {
	await wasm_bindgen();
	formInit();
}

function updateHTML(status, result) {
	const inferStatus = document.getElementById("infer-status");
	const resultList = document.getElementById("result-list");
	inferStatus.textContent = status;
	resultList.innerHTML = result;
}

function formInit() {
	const form = document.getElementById("url-form");

	form.addEventListener("submit", async (event) => {
		updateHTML("推論中...", "");

		event.preventDefault();
		const url = document.getElementById("image-url").value;

		try {
			// string: [{ label: string, score: number, rank: number }, ...]
			const top5 = await analyzeImageOnWasm(url);

			updateHTML("推論完了", JSON.parse(top5)
				.map((item) => {
					return `<li>${item.label}: ${item.score}</li>`;
				})
				.join("")
			);
		} catch (error) {
			console.error(error);

			updateHTML("推論失敗", "");
		}
	});
}

init();
form_init();
