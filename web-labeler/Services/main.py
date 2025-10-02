from fastapi import FastAPI, File, UploadFile, HTTPException
import tempfile, os
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# Import the commons functions and parameters from your commons.py
from commons import (
    preprocess_data_without_scaling,
    time_column,
    target_column,
    timesteps, load_data
)

app = FastAPI()

# Add CORS middleware (make sure this is done before your endpoints are defined)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # allow your React client origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to hold the processed DataFrame
processed_data = None


@app.post("/update_energy")
def update_energy(time: str, new_value: float):
    """
    Updates the energy value for a given time stamp.
    Example call: /update_energy?time=2023-09-14&new_value=123.45
    """
    global processed_data
    if processed_data is None:
        raise HTTPException(status_code=404, detail="No processed data available.")
    try:
        dt = pd.to_datetime(time)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid time format")

    mask = processed_data[time_column] == dt
    if not mask.any():
        raise HTTPException(status_code=404, detail="Time value not found")

    processed_data.loc[mask, "Inv 1 AC-Leistung (W)"] = new_value
    return {"message": "Energy value updated successfully."}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Accepts a CSV file, loads it using load_data from commons, and sets the global variable.
    """
    global processed_data
    try:
        # Create a temporary file to hold the uploaded CSV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Load the CSV file using load_data from commons
        processed_data = load_data(file_path=tmp_path, time_column=time_column)
        # Only add anomaly_label column if not present
        if "anomaly_label" not in processed_data.columns:
            processed_data["anomaly_label"] = 1

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {"message": "File uploaded and loaded successfully."}

@app.post("/uploadandpreprocess")
async def upload_and_preprocess_file(file: UploadFile = File(...)):
    """
    Accepts a CSV file, processes it using commons.py and adds an anomaly_label column.
    """
    global processed_data
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        processed_data = preprocess_data_without_scaling(
            file_path=tmp_path,
            time_column=time_column,
            target_column=target_column,
            timesteps=timesteps,
            drop_time_column=False
        )

        if "anomaly_label" not in processed_data.columns:
            processed_data["anomaly_label"] = 1
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {"message": "File uploaded and processed successfully."}


@app.get("/data")
def get_data():
    """
    Returns the processed data as JSON.
    """
    if processed_data is None:
        raise HTTPException(status_code=404, detail="No processed data available.")
    return processed_data.to_dict(orient="records")


@app.post("/update_anomaly")
def update_anomaly(time: str, anomaly_label: int):
    """
    Updates the anomaly_label for a given time value.
    Example: { "time": "2023-09-14", "anomaly_label": -1 }
    """
    global processed_data
    if processed_data is None:
        raise HTTPException(status_code=404, detail="No processed data available.")
    try:
        dt = pd.to_datetime(time)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid time format")
    mask = processed_data[time_column] == dt
    if not mask.any():
        raise HTTPException(status_code=404, detail="Time value not found")
    processed_data.loc[mask, "anomaly_label"] = anomaly_label
    return {"message": "Anomaly updated successfully."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
