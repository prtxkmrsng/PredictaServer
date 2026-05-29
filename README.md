# PredictaServer 🚀

An enterprise-ready, high-throughput time-series API engine engineered for cloud infrastructure capacity forecasting and proactive horizontal auto-scaling. The system ingests running performance counters, processes metrics across a deep, multi-layered Long Short-Term Memory (LSTM) recurrent neural network, and outputs predictive resource scheduling directives without blocking the main network request loop.

---

## 🏗️ Architectural Topology

- **Deep Learning Core:** Multi-layered Keras LSTM (Long Short-Term Memory) sequential stack built with custom Dropout layers to mitigate model co-adaptation and overfitting.
- **ASGI Production Interface:** FastAPI serving structured inputs mapped through robust type-safe Pydantic data validation constraints.
- [cite_start]**High-Throughput Concurrency:** Integrated `asyncio.to_thread` worker pool routing blocking mathematical matrix inferences onto external threads to prevent event loop delay spikes[cite: 107].
- [cite_start]**Ecosystem Isolation:** Containerized deployment layout managed entirely through thin Docker base structures to ensure reproducible cross-platform execution.

---

## 🛠️ Project Structure

```text
PredictaServer/
│
├── .gitattributes              # Line-ending normalization rules
├── .gitignore                  # Active tracking exemptions map
├── Dockerfile                  # Container isolation execution manual
├── main.py                     # Non-blocking FastAPI inference system
├── requirements.txt            # System dependencies snapshot blueprint
├── server_lstm_model.keras     # Keras model for inference
└── train_lstm.py               # Deep LSTM training and validation pipeline
```

## 🚀 Local Windows Setup Instruction
### 1. Prerequisite Checklist
- **Python:** Ensure Python 3.11 is installed locally.  
- **Environment Engine:** Install and run Docker Desktop configured with the native WSL2 execution backend.
### 2. Native Model Preparation:


Clone this repository, navigate to your root directory, and set up your virtual space:

``Skip training if you have`` ***server_lstm_model.keras*** ``downloaded.``

To train your network parameters locally outside a container configuration, configure your environment space:

```
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
```

Train and serialize your high-performance deep memory matrix layer weights:
```
python train_lstm.py
```

This generates your validation loss calculations and creates ***server_lstm_model.keras***.

### 3. Containerized Ecosystem Compilation:

With your model parameters saved/downloaded, leverage Docker to construct your portable workspace node:  

```
# Compile the secure Docker image layer
docker build -t predictaserver:v1 .

# Launch the decoupled background network engine node
docker run -d -p 8000:8000 --name predicta_node predictaserver:v1
```

### 4. Interactive Execution Testing:
Open your browser and navigate to:
- Interactive OpenAPI Documentation Swagger UI: http://localhost:8000/docs

You can verify the container endpoint by passing this mock rising load metrics JSON to the ``/predict`` controller route:

```
JSON{
  "sequence": [0.55, 0.58, 0.61, 0.65, 0.70, 0.73, 0.76, 0.78, 0.81, 0.83]
}
```