# PredictaServer 🚀

An enterprise-ready, high-throughput time-series API engine engineered for cloud infrastructure capacity forecasting and proactive horizontal auto-scaling. The system ingests running performance counters, processes metrics across a deep, multi-layered Long Short-Term Memory (LSTM) recurrent neural network, and outputs predictive resource scheduling directives without blocking the main network request loop.

---

## 🏗️ Architectural Topology

- **Deep Learning Core:** Multi-layered Keras LSTM (Long Short-Term Memory) sequential stack built with custom Dropout layers to mitigate model co-adaptation and overfitting.
- **ASGI Production Interface:** FastAPI serving structured inputs mapped through robust type-safe Pydantic data validation constraints.
- **High-Throughput Concurrency:** Integrated `asyncio.to_thread` worker pool routing blocking mathematical matrix inferences onto external threads to prevent event loop delay spikes.
- **Ecosystem Isolation:** Containerized deployment layout managed entirely through thin Docker base structures to ensure reproducible cross-platform execution.
- **Global Data Scaling:** Utilizes a globally fitted `MinMaxScaler` serialized via `joblib` during training to ensure real-time inference inputs are mathematically evaluated against the true historical limits of the training data.

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
├── scaler.gz                   # Serialized global scaling boundaries
└── train_lstm.py               # Deep LSTM training and validation pipeline

```

## 🚀 Local Windows Setup Instruction

### 1. Prerequisite Checklist

- **Python:** Ensure Python 3.11 is installed locally.
- **Environment Engine:** Install and run Docker Desktop configured with the native WSL2 execution backend.

### 2. Native Model Preparation:

Clone this repository, navigate to your root directory, and set up your virtual space:

`Skip training if you have` **_server_lstm_model.keras_** `and` **_scaler.gz_** `downloaded.`

- To train your network parameters locally outside a container configuration, you will need WSL installed with Ubuntu.
- Once you have your WSL shell configured, setup your virtual environment:

```bash
sudo apt update # (Updates your package list; type your password when asked)
sudo apt install python3-pip python3-venv -y # (Installs Python tools)
mkdir predictaserver_project # (Creates a folder)
cd predictaserver_project # (Moves inside the folder)
python3 -m venv venv # (Creates an isolated virtual environment)
source venv/bin/activate # (Activates the environment)
pip install jupyter pandas tensorflow scikit-learn # (Installs the required deep learning libraries)
```

- Copy the directory `PredictaServer_Linux/` into our new folder `predictaserver_project/`.
- Instantiate our jupyter notebook:

```bash
jupyter notebook trainer.ipynb
```

- Run all the cells of this notebook.
- This generates your validation loss calculations and creates **_server_lstm_model.keras_** and **_scaler.gz_**.
- Copy these back to your original project root directory.

### 3. Containerized Ecosystem Compilation:

With your model parameters saved/downloaded, leverage Docker to construct your portable workspace node:

```bash
# Compile the secure Docker image layer
docker build -t predictaserver:v1 .

# Launch the decoupled background network engine node
docker run -d -p 8000:8000 --name predicta_node predictaserver:v1
```

### 4. Interactive Execution Testing:

Open your browser and navigate to:

- Interactive OpenAPI Documentation Swagger UI: http://localhost:8000/docs

You can verify the container endpoint by passing this mock rising load metrics JSON to the `/predict` controller route:

**Request Payload:**

```json
{
  "sequence": [0.55, 0.58, 0.61, 0.65, 0.7, 0.73, 0.76, 0.78, 0.81, 0.83]
}
```

**Expected Response Payload:**

```json
{
  "current_trend_average": 70.0,
  "forecasted_utilization": 86.2,
  "auto_scale_trigger": true,
  "recommended_instances": 6
}
```
