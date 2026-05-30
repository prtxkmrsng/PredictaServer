import asyncio
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Initialize FastAPI instance
app = FastAPI(
    title="PredictaServer | Enterprise Scaling Engine",
    description="High-performance sequential deep learning service for infrastructure utilization forecasting.",
    version="1.0.0"
)

# Global AI assets
MODEL_PATH = "server_lstm_model.keras"
model = None
scaler = MinMaxScaler(feature_range=(0, 1))

@app.on_event("startup")
async def load_serialized_model():
    """
    Loads our binary Keras neural network into memory on API launch.
    """
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[Startup] Successfully initialized deep learning checkpoint: {MODEL_PATH}")
    except Exception as e:
        print(f"[Critical Startup Error] Failed to serialize model: {e}")
        # We don't crash the server immediately, but keep model as None to handle gracefully

# -------------------------------------------------------------------
# DATA VALIDATION STRUCTURES
# -------------------------------------------------------------------
class UtilizationPayload(BaseModel):
    # Enforce exactly 10 sequential steps matching our LSTM window architecture
    sequence: list[float] = Field(
        ..., 
        example=[0.134, 0.134, 0.136, 0.132, 0.201, 0.350, 0.500, 0.800, 1.200, 1.500],
        description="The past 10 continuous data readings representing scaled server CPU metrics."
    )

class AllocationResponse(BaseModel):
    current_trend_average: float
    forecasted_utilization: float
    auto_scale_trigger: bool
    recommended_instances: int

# -------------------------------------------------------------------
# ASYNC MODEL EXECUTION WRAPPER
# -------------------------------------------------------------------
def sync_model_inference(input_sequence: list[float]) -> float:
    """
    Isolated pure synchronous matrix evaluation.
    This runs inside a background worker thread.
    """
    # 1. Shape the raw data into a 2D array for the scaler
    raw_array = np.array(input_sequence).reshape(-1, 1)
    
    # 2. Scale the data to the 0-1 range the neural network expects
    scaled_array = scaler.fit_transform(raw_array)
    
    # 3. Reshape into the 3D Tensor matrix shape: (batch_size, timesteps, features)
    inference_batch = scaled_array.reshape(1, 10, 1)
    
    # 4. Execute the prediction
    raw_prediction = model.predict(inference_batch, verbose=0)
    
    # 5. Inverse transform to convert the 0-1 output back into real-world CPU values
    real_value_prediction = scaler.inverse_transform(raw_prediction)
    
    return float(real_value_prediction[0][0])

# -------------------------------------------------------------------
# ROUTING CONTROLLERS
# -------------------------------------------------------------------
@app.post("/predict", response_model=AllocationResponse, status_code=status.HTTP_200_OK)
async def predict_infrastructure_scaling(payload: UtilizationPayload):
    """
    Consumes sequential infrastructure metrics, handles multi-threaded prediction, 
    and outputs predictive scaling decisions.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Neural network engine is uninitialized or unreadable."
        )
        
    # Validate payload array matrix sizing constraints
    if len(payload.sequence) != 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid matrix dimensional length. Expected exactly 10 sequential elements, got {len(payload.sequence)}."
        )
    
    try:
        # Offload the blocking model prediction execution onto an isolated background thread
        forecasted_value = await asyncio.to_thread(sync_model_inference, payload.sequence)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference computation cluster failure: {str(e)}"
        )
        
    # Decision matrix logic for horizontal auto-scaling actions
    mean_historical = float(np.mean(payload.sequence))
    trigger_action = False
    instance_allocation = 2
    
    if forecasted_value >= 0.80:
        trigger_action = True
        instance_allocation = 6
    elif forecasted_value >= 0.65:
        trigger_action = True
        instance_allocation = 4
        
    return AllocationResponse(
        current_trend_average=round(mean_historical * 100, 2),
        forecasted_utilization=round(forecasted_value * 100, 2),
        auto_scale_trigger=trigger_action,
        recommended_instances=instance_allocation
    )

@app.get("/health")
def health_check():
    return {"status": "healthy", "engine_loaded": model is not None}