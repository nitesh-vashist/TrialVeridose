import os
import logging
import json
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator, conint, confloat, constr
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import numpy as np
import pandas as pd

# --- Setup Logging ---
DEFAULT_LOG_CONFIG = { "level": "INFO", "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s" }
log_conf = DEFAULT_LOG_CONFIG

log_level = log_conf.get('level', 'INFO').upper()
log_format = log_conf.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format=log_format)
logger = logging.getLogger(__name__)


# --- Import Prediction Logic ---
try:
    from predict_fake_report import predict_report_anomaly, parse_list_string
    PREDICTION_LOADED = True
    logger.info("Successfully imported prediction logic from predict_fake_report.py")
except ImportError as e:
    logger.error(f"Failed to import from predict_fake_report.py: {e}", exc_info=True)
    PREDICTION_LOADED = False
    def predict_report_anomaly(record_dict):
        return {"error": "Prediction function not loaded due to import error", "rule_score": 0, "triggered_rules": [], "model_predictions": {}, "final_combined_prediction": {"label": "Error", "numeric": 0}}
    def parse_list_string(s): return []
except Exception as e:
    logger.error(f"Error during initial import/setup from predict_fake_report.py: {e}", exc_info=True)
    PREDICTION_LOADED = False
    def predict_report_anomaly(record_dict):
        return {"error": f"Prediction function not loaded due to setup error: {e}", "rule_score": 0, "triggered_rules": [], "model_predictions": {}, "final_combined_prediction": {"label": "Error", "numeric": 0}}
    def parse_list_string(s): return []


# --- Enums for Validation ---
class GenderEnum(str, Enum):
    male = 'Male'
    female = 'Female'
    other = 'Other'
    unknown = 'Unknown' # Allow Unknown explicitly

class SideEffectSeverityEnum(str, Enum):
    none = 'None'
    mild = 'Mild'
    moderate = 'Moderate'
    severe = 'Severe'
    unknown = 'Unknown'

class OverallHealthStatusEnum(str, Enum):
    improved = 'Improved'
    same = 'Same'
    worse = 'Worse'
    unknown = 'Unknown'


# --- Pydantic Models for Request and Response (with enhanced validation) ---

class PatientRecord(BaseModel):
    """Defines the structure for a single patient's data in the request."""
    patient_id: constr(min_length=1) # Ensure patient_id is not empty
    age: Optional[confloat(ge=0, le=130)] = None # Realistic age range
    trialDuration: Optional[confloat(ge=0)] = Field(None, alias='trialDuration') # Duration >= 0
    dosage_numeric: Optional[confloat(ge=0)] = Field(None, alias='dosage_numeric') # Dosage >= 0
    drugName: Optional[constr(min_length=1)] = Field(None, alias='drugName')
    doctorNotes: Optional[str] = Field(None, alias='doctorNotes') # Keep as optional string
    gender: Optional[GenderEnum] = None # Use Enum for validation
    sideEffectSeverity: Optional[SideEffectSeverityEnum] = Field(None, alias='sideEffectSeverity') # Use Enum
    overallHealthStatus: Optional[OverallHealthStatusEnum] = Field(None, alias='overallHealthStatus') # Use Enum
    symptomImprovementScore: Optional[confloat(ge=0, le=10)] = Field(None, alias='symptomImprovementScore') # Score 0-10
    knownAllergies: Optional[Union[List[str], str]] = Field(None, alias='knownAllergies')
    conditions_during_encounter: Optional[Union[List[str], str]] = Field(None, alias='conditions_during_encounter')
    new_conditions_after_med_start: Optional[Union[List[str], str]] = Field(None, alias='new_conditions_after_med_start')
    trialSideEffects: Optional[Union[List[str], str]] = Field(None, alias='trialSideEffects')
    has_impossible_observation: Optional[bool] = Field(None, alias='has_impossible_observation')

    # Include other optional fields used by predict_report_anomaly if available in input
    # The prediction script has defaults, but providing them might be more accurate
    proc_count: Optional[int] = None
    # Optional fields with basic type hints
    proc_count: Optional[int] = Field(None, ge=0)
    claim_count: Optional[int] = Field(None, ge=0)
    avg_temp_during_trial: Optional[float] = None
    max_hr_during_trial: Optional[float] = Field(None, ge=0)
    min_bp_systolic_during_trial: Optional[float] = Field(None, ge=0)
    max_bp_systolic_during_trial: Optional[float] = Field(None, ge=0)
    min_bp_diastolic_during_trial: Optional[float] = Field(None, ge=0)
    max_bp_diastolic_during_trial: Optional[float] = Field(None, ge=0)
    count_side_effect_keywords_obs: Optional[int] = Field(None, ge=0)
    count_improvement_keywords_obs: Optional[int] = Field(None, ge=0)
    count_worsening_keywords_obs: Optional[int] = Field(None, ge=0)
    avg_condition_duration: Optional[float] = Field(None, ge=0)
    std_temp_during_trial: Optional[float] = Field(None, ge=0)
    std_hr_during_trial: Optional[float] = Field(None, ge=0)
    std_bp_systolic_during_trial: Optional[float] = Field(None, ge=0)
    std_bp_diastolic_during_trial: Optional[float] = Field(None, ge=0)
    nlp_verb_count: Optional[int] = Field(None, ge=0)
    nlp_noun_count: Optional[int] = Field(None, ge=0)
    nlp_adj_count: Optional[int] = Field(None, ge=0)
    nlp_ner_count: Optional[int] = Field(None, ge=0)

    # Validator to ensure list-like fields are handled correctly if passed as strings
    @validator('knownAllergies', 'conditions_during_encounter', 'new_conditions_after_med_start', 'trialSideEffects', pre=True, always=True)
    def ensure_list_format(cls, v):
        if isinstance(v, str):
            return parse_list_string(v)
        # Allow None or existing lists
        return v if v is not None else []

    class Config:
        allow_population_by_field_name = True # Allows using aliases like 'trialDuration'

class HospitalPredictionRequest(BaseModel):
    """Defines the structure of the incoming API request."""
    hospital_id: constr(min_length=1)
    patient_data: List[PatientRecord] = Field(..., min_items=1) # Ensure patient_data is not empty

class IndividualPredictionResult(BaseModel):
    """Structure for the result of a single patient prediction, including potential errors."""
    patient_id: str
    rule_score: Optional[int] = None # Make optional to accommodate processing errors
    triggered_rules: Optional[List[str]] = None
    model_predictions: Optional[Dict[str, Dict[str, Any]]] = None
    final_combined_prediction: Optional[Dict[str, Union[str, int, None]]] = None # Allow None in numeric
    error: Optional[str] = None # Field to hold processing errors

class OverallHospitalPrediction(BaseModel):
    """Structure for the overall hospital-level prediction."""
    label: str
    numeric: int
    anomaly_count: int
    normal_count: int
    error_count: int
    total_patients: int
    anomaly_percentage: float
    aggregated_triggered_rules: List[str] # Added field for unique rules

class HospitalPredictionResponse(BaseModel):
    """Defines the structure of the API response."""
    hospital_id: str
    individual_patient_results: List[IndividualPredictionResult]
    overall_hospital_prediction: OverallHospitalPrediction


# --- FastAPI App ---
app = FastAPI(
    title="Hospital Anomaly Prediction API",
    description="Predicts anomalies in patient data for a given hospital ID based on individual patient records and rules/models.",
    version="1.1.0"
)

# --- Custom Exception Handler for Validation Errors ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Input validation failed: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )

# --- API Endpoint ---
@app.post("/predict_hospital_anomaly",
          response_model=HospitalPredictionResponse,
          summary="Predict Hospital Anomaly",
          description="Accepts hospital ID and a list of patient data records. Returns individual predictions and an overall hospital-level prediction.",
          tags=["Prediction"])
async def predict_hospital_anomaly_endpoint(request: HospitalPredictionRequest):
    """
    Processes a batch of patient records for a given hospital ID:
    - Validates input data.
    - Runs each patient record through the anomaly prediction model.
    - Aggregates results to provide an overall hospital prediction.
    """
    # Check if prediction logic loaded correctly during startup
    if not PREDICTION_LOADED:
        logger.error("Prediction endpoint called, but prediction logic failed to load.")
        raise HTTPException(status_code=500, detail="Internal server error: Prediction model not available.")

    """
    Accepts hospital ID and a list of patient data records.
    Returns individual predictions for each patient and an overall
    hospital-level prediction based on majority voting.
    """
    hospital_id = request.hospital_id
    patient_data_list = request.patient_data # Already validated by Pydantic

    logger.info(f"Received prediction request for hospital: {hospital_id} with {len(patient_data_list)} patients.")

    individual_results: List[IndividualPredictionResult] = []
    anomaly_count = 0
    normal_count = 0
    error_count = 0
    total_patients = len(patient_data_list)
    all_triggered_rules = set()

    for patient_record_model in patient_data_list:
        patient_id = patient_record_model.patient_id
        logger.debug(f"Processing patient: {patient_id}")
        # Convert Pydantic model to dictionary for the prediction function
        # Use exclude_none=True to avoid passing explicit None values unless necessary
        patient_dict = patient_record_model.model_dump(by_alias=True, exclude_none=True)

        try:
            # Call the prediction function from the imported script
            prediction_output = predict_report_anomaly(patient_dict)

            # Check if the prediction function itself returned an error
            if prediction_output.get("error"):
                logger.warning(f"Prediction function returned error for patient {patient_id}: {prediction_output['error']}")
                result_data = {
                    "patient_id": patient_id,
                    "error": prediction_output["error"]
                    # Other fields will be None by default in Pydantic model
                }
                error_count += 1
            else:
                # Successful prediction
                result_data = {
                    "patient_id": patient_id,
                    **prediction_output # Unpack the prediction output dictionary
                }
                 # Add triggered rules to the set
                if result_data.get("triggered_rules"):
                    all_triggered_rules.update(result_data["triggered_rules"])
                # Tally counts for overall prediction
                final_pred = result_data.get("final_combined_prediction", {})
                if final_pred.get("numeric") == -1:
                    anomaly_count += 1
                elif final_pred.get("numeric") == 1:
                    normal_count += 1
                else: # Includes cases where numeric is 0 or None (error state)
                    error_count += 1 # Count cases where prediction is 'Error' as errors

            # Create and append the result object
            individual_result = IndividualPredictionResult(**result_data)
            individual_results.append(individual_result)

        except Exception as e:
            # Catch unexpected errors during the call or processing the result
            logger.error(f"Unexpected error processing patient {patient_id}: {e}", exc_info=True)
            error_result = IndividualPredictionResult(
                patient_id=patient_id,
                error=f"Unexpected server error: {e}"
            )
            individual_results.append(error_result)
            error_count += 1

    # Determine overall hospital prediction (majority vote among non-error results)
    valid_predictions = total_patients - error_count # Count only successful predictions
    overall_label = "Error" # Default if all errored or no valid predictions
    overall_numeric = 0     # 0 for Error/Undetermined

    if valid_predictions > 0:
        # Base decision only on successfully processed patients
        if anomaly_count > (valid_predictions / 2.0):
            overall_label = "Outlier/Anomaly"
            overall_numeric = -1
        else: # Includes cases where anomaly_count <= valid_predictions / 2.0
            overall_label = "Inlier/Normal"
            overall_numeric = 1
        anomaly_percentage = (anomaly_count / valid_predictions * 100)
    else:
        # Handle cases: all patients errored, or input list was empty (though caught earlier)
        anomaly_percentage = 0.0
        if total_patients > 0:
             logger.warning(f"Hospital {hospital_id}: All {total_patients} patient records resulted in processing errors.")
        # Keep label "Error", numeric 0

    logger.info(f"Hospital {hospital_id}: Overall prediction: {overall_label} ({anomaly_count} Anomaly, {normal_count} Normal, {error_count} Error)")

    overall_prediction = OverallHospitalPrediction(
        label=overall_label,
        numeric=overall_numeric,
        anomaly_count=anomaly_count,
        normal_count=normal_count,
        error_count=error_count,
        total_patients=total_patients,
        anomaly_percentage=round(anomaly_percentage, 2),
        aggregated_triggered_rules=sorted(list(all_triggered_rules)) # Convert set to sorted list
    )

    # Construct and return the final response
    response = HospitalPredictionResponse(
        hospital_id=hospital_id,
        individual_patient_results=individual_results,
        overall_hospital_prediction=overall_prediction
    )

    return response

# --- Run the server (for local development) ---
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    reload_flag = os.getenv("RELOAD", "false").lower() == "true"

    logger.info(f"Starting FastAPI server on {host}:{port} (Reload: {reload_flag})")
    logger.info(f"Access API documentation at http://{host}:{port}/docs")
    uvicorn.run("api_server:app", host=host, port=port, reload=reload_flag)
