const { startup } = wasm_bindgen;

let worker;

async function init() {
	await wasm_bindgen();
	startup();
	
	// Create and setup worker
	worker = new Worker('./worker.js');
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
			// Send image URL to worker
			worker.postMessage(url);
			
			// Setup one-time message handler for this request
			const messageHandler = (event) => {
				worker.removeEventListener('message', messageHandler);
				
				try {
					const top5 = event.data;
					const result = JSON.parse(top5);
					
					updateHTML("推論完了", result
						.map((item) => {
							return `<li>${item.label}: ${item.score}</li>`;
						})
						.join("")
					);
				} catch (error) {
					console.error('Failed to parse worker response:', error);
					updateHTML("推論失敗", "");
				}
			};
			
			worker.addEventListener('message', messageHandler);
			
			// Handle worker errors
			const errorHandler = (error) => {
				worker.removeEventListener('error', errorHandler);
				console.error('Worker error:', error);
				updateHTML("推論失敗", "");
			};
			
			worker.addEventListener('error', errorHandler);

		} catch (error) {
			console.error(error);
			updateHTML("推論失敗", "");
		}
	});
}

init();
