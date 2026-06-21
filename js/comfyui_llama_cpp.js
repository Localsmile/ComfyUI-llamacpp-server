import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

app.registerExtension({
    name: "comfyui-llama-cpp",
    async nodeCreated(node) {
        if (node.comfyClass === "LlamaCppServerNode") {
            // Check if Reconnect button already exists
            const hasBtn = node.widgets && node.widgets.some(w => w.name === "reconnect" || w.label === "Reconnect");
            if (!hasBtn) {
                node.addWidget("button", "Reconnect", "reconnect", () => {
                    reconnectServer(node);
                });
            }
        }
    }
});

async function reconnectServer(node) {
    const urlWidget = node.widgets.find(w => w.name === "url");
    const modelWidget = node.widgets.find(w => w.name === "model");
    if (!urlWidget || !modelWidget) return;
    
    const url = urlWidget.value;
    try {
        modelWidget.options.values = ["Connecting..."];
        modelWidget.value = "Connecting...";
        node.setSize(node.size);
        app.canvas.draw(true, true);
        
        const response = await api.fetchApi("/llama-cpp/models", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ url })
        });
        
        if (!response.ok) {
            throw new Error("Failed to connect to llama.cpp server");
        }
        
        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }
        
        const models = data.models || [];
        const options = ["Auto (Detect Active Model)", ...models.filter(m => m !== "Auto (Detect Active Model)")];
        
        modelWidget.options.values = options;
        const activeModel = data.active_model;
        if (activeModel && options.includes(activeModel)) {
            modelWidget.value = activeModel;
        } else {
            modelWidget.value = "Auto (Detect Active Model)";
        }
    } catch (err) {
        console.error(err);
        modelWidget.options.values = [`Error: ${err.message}`];
        modelWidget.value = `Error: ${err.message}`;
    }
    app.canvas.draw(true, true);
}
