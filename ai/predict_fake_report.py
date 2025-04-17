import pandas as pd
import numpy as np
import joblib
import ast
import warnings
import json
import os
import logging 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json

# --- Configuration Loading ---
config = {
  "rule_thresholds": { "min_age": 0, "max_age": 120, "min_trial_duration": 1, "max_trial_duration_warning": 1825 },
  "logging": { "level": "INFO", "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s" }
}

# --- Logging Setup ---
log_level = config['logging'].get('level', 'INFO').upper()
log_format = config['logging'].get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format=log_format)
logger = logging.getLogger(__name__)

# Suppress warnings and TF logs
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# --- Constants ---
ARTIFACTS_FILE = os.getenv('ARTIFACTS_FILE', 'model_artifacts_tuned.joblib')
RULE_THRESHOLDS = config['rule_thresholds']

# --- Helper Functions ---
def parse_list_string(s):
    """Safely parses a string representation of a list or returns list if input is list."""
    try:
        if pd.isna(s): return []
        if isinstance(s, list): return s # Already a list
        if isinstance(s, str):
            # Handle simple comma-separated string if not a valid list literal
            if not (s.startswith('[') and s.endswith(']')):
                 s_cleaned = s.strip().strip("'\"")
                 # Return list of strings, handle empty strings after split
                 return [item.strip() for item in s_cleaned.split(',') if item.strip()]
            # If it looks like a list literal, try evaluating safely
            parsed_list = ast.literal_eval(s)
            if isinstance(parsed_list, list): return parsed_list
            elif isinstance(parsed_list, tuple): return list(parsed_list)
            else: return [str(parsed_list)] # Wrap single items
        return [] # Fallback for other types
    except (ValueError, SyntaxError, TypeError):
        # Fallback for strings that look like lists but aren't valid literals
        if isinstance(s, str) and s.strip():
             s_cleaned = s.strip().strip('[]\'"')
             return [item.strip() for item in s_cleaned.split(',') if item.strip()]
        return []

# --- Sentiment Analysis Function (copied from training script) ---
SIA = SentimentIntensityAnalyzer()
def get_sentiment_score(text):
    """Calculates the VADER compound sentiment score."""
    if pd.isna(text):
        return 0.0 # Neutral score for missing text
    try:
        # Ensure text is string
        text = str(text)
        vs = SIA.polarity_scores(text)
        return vs['compound']
    except Exception:
         # Handle potential errors during analysis
         return 0.0

# --- 1. Load Artifacts ---
logger.info(f"Loading model artifacts from {ARTIFACTS_FILE}...")
try:
    artifacts = joblib.load(ARTIFACTS_FILE)
    preprocessor_artifact = artifacts['preprocessor']
    ct = preprocessor_artifact['col_transformer']
    mlb_transformers = preprocessor_artifact['mlb_transformers']
    mlb_classes = preprocessor_artifact.get('mlb_classes', {})
    dosage_stats = preprocessor_artifact.get('dosage_stats', {}) # Load dosage stats
    drug_freq_map = preprocessor_artifact.get('drug_freq_map', {}) # Load drug freq map
    models = artifacts['models']
    # Load Autoencoder path and threshold
    autoencoder_model_path = artifacts.get('autoencoder_model_path') # Path stored during training
    autoencoder_threshold = artifacts.get('autoencoder_threshold') # Threshold stored during training
    autoencoder = None

    # Improved AE loading check
    if autoencoder_model_path and isinstance(autoencoder_model_path, str) and autoencoder_threshold is not None:
        if os.path.exists(autoencoder_model_path):
            try:
                autoencoder = keras.models.load_model(autoencoder_model_path)
                logger.info(f"Autoencoder model loaded successfully from {autoencoder_model_path}.")
            except Exception as ae_load_err:
                logger.warning(f"Error loading Autoencoder model from {autoencoder_model_path}: {ae_load_err}", exc_info=True)
                autoencoder = None # Ensure it's None if loading fails
        else:
            logger.warning(f"Autoencoder model file not found at path specified in artifacts: {autoencoder_model_path}")
            autoencoder = None
    else:
         logger.warning(f"Autoencoder model path ('{autoencoder_model_path}') or threshold ('{autoencoder_threshold}') not found or invalid in artifacts.")
         autoencoder = None # Ensure AE is None if path/threshold missing or invalid

    logger.info("Artifacts (preprocessor, models, stats, maps, AE path, AE threshold) loaded successfully.")
except FileNotFoundError:
    logger.error(f"Artifacts file not found - {ARTIFACTS_FILE}. Run train_anomaly_model.py first.", exc_info=True)
    # Raise exception instead of exit() for better handling in API
    raise FileNotFoundError(f"Artifacts file not found: {ARTIFACTS_FILE}")
except KeyError as e:
    logger.error(f"Missing key in artifact file '{ARTIFACTS_FILE}': {e}. Ensure training script saves all components.", exc_info=True)
    raise KeyError(f"Missing key in artifact file '{ARTIFACTS_FILE}': {e}")
except Exception as e:
    logger.error(f"Error loading artifacts: {e}", exc_info=True)
    raise RuntimeError(f"Error loading artifacts: {e}")

# --- Define Feature Lists (Matching Training Script V3 + New Features) ---
# These should align with the features *after* engineering in the training script
numerical_features = [
    'age', 'trialDuration', 'dosage_numeric', 'drugName_freq',
    'noteLength', 'noteWordCount', 'proc_count', 'claim_count',
    'dosage_zscore_overall',
    'avg_temp_during_trial', 'max_hr_during_trial',
    'min_bp_systolic_during_trial', 'max_bp_systolic_during_trial',
    'min_bp_diastolic_during_trial', 'max_bp_diastolic_during_trial',
    'count_side_effect_keywords_obs', 'count_improvement_keywords_obs', 'count_worsening_keywords_obs',
    # Added features from feature engineering step
    'avg_condition_duration', 'std_temp_during_trial', 'std_hr_during_trial',
    'std_bp_systolic_during_trial', 'std_bp_diastolic_during_trial',
    # Added NLP features
    'nlp_verb_count', 'nlp_noun_count', 'nlp_adj_count', 'nlp_ner_count',
    # Add sentiment score
    'sentiment_score',
    # Add interaction feature
    'dosage_per_age'
]
categorical_features = [
    'gender',
    'sideEffectSeverity', 'overallHealthStatus', 'symptomImprovementScore',
    'isDosageOutOfRange_overall',
    'has_impossible_observation'
]
text_feature = 'doctorNotes'
multilabel_features = [
    'knownAllergies', 'conditions_during_encounter',
    'new_conditions_after_med_start', 'trialSideEffects'
]

# Get the exact feature list expected by the loaded ColumnTransformer for non-MLB features
try:
    ct_expected_features = []
    raw_ct_features = []
    for name, transformer_obj, columns in ct.transformers_:
        if transformer_obj != 'drop' and transformer_obj != 'passthrough':
             raw_ct_features.extend(columns)

    # Robustness: Filter out likely erroneous single-character features from TF-IDF
    ct_expected_features = [f for f in raw_ct_features if len(f) > 1 or f.isalnum()] # Keep multi-char or single digit/letter
    removed_features = set(raw_ct_features) - set(ct_expected_features)
    if removed_features:
        logger.warning(f"Removed potential erroneous single-char features from expected list: {sorted(list(removed_features))}")

    logger.debug(f"Loaded ColumnTransformer expects features: {ct_expected_features}")

except Exception as e:
    logger.warning(f"Could not reliably get features from loaded ColumnTransformer: {e}. Prediction might fail.", exc_info=True)
    ct_expected_features = [] # Cannot proceed reliably without knowing expected features

# --- 2. Define Rule-Based Checks ---
def check_rules(record, thresholds):
    """Applies predefined rules to the input record (using V3 feature names)."""
    rule_score = 0
    triggered_rules = []

    min_age = thresholds.get('min_age', 0)
    max_age = thresholds.get('max_age', 120)
    min_duration = thresholds.get('min_trial_duration', 1)
    max_duration_warn = thresholds.get('max_trial_duration_warning', 1825)

    # Rule 1: Age out of range
    try: age = float(record.get('age', np.nan))
    except (ValueError, TypeError): age = np.nan
    if pd.notna(age) and not (min_age <= age <= max_age):
        rule_score += 1; triggered_rules.append(f"Age out of range ({min_age}-{max_age})")

    # Rule 2: Trial duration invalid
    try: duration = float(record.get('trialDuration', np.nan))
    except (ValueError, TypeError): duration = np.nan
    if pd.notna(duration) and duration < min_duration:
        rule_score += 1; triggered_rules.append(f"Trial duration < {min_duration}")

    # Rule 3: Severity vs Side Effects mismatch (using trialSideEffects)
    severity = record.get('sideEffectSeverity', 'None')
    effects = record.get('trialSideEffects', [])
    if not isinstance(effects, list): effects = parse_list_string(effects) # Ensure list
    if severity in ['Severe', 'Moderate'] and (not effects or effects == ['None'] or not any(e != "Observation Keyword Trigger" for e in effects)): # Check if only trigger or empty
        rule_score += 1; triggered_rules.append("Severity is Moderate/Severe but no specific trial side effects listed")

    # Rule 4: Illogical Score vs Status
    try: score = float(record.get('symptomImprovementScore', 5))
    except (ValueError, TypeError): score = 5
    status = record.get('overallHealthStatus', 'Same')
    if score >= 8 and status == 'Worse':
        rule_score += 1; triggered_rules.append("High symptom improvement score but overall status is Worse")
    if score <= 2 and status == 'Improved':
        rule_score += 1; triggered_rules.append("Low symptom improvement score but overall status is Improved")

    # Rule 5: Gender vs Condition (using conditions_during_encounter)
    gender = record.get('gender', 'Unknown')
    conditions = record.get('conditions_during_encounter', [])
    if not isinstance(conditions, list): conditions = parse_list_string(conditions)
    pregnancy_conditions = {'Normal Pregnancy', 'Antepartum Eclampsia', 'Preeclampsia', 'Tubal Pregnancy'}
    if gender == 'Male' and any(cond in pregnancy_conditions for cond in conditions):
         rule_score += 1; triggered_rules.append("Pregnancy-related condition listed for Male patient")

    # Rule 6: Unusually Long Trial Duration (Warning Rule)
    if pd.notna(duration) and duration > max_duration_warn:
        # This rule might just be a warning, not increase score, depending on requirements
        # rule_score += 1 # Optional: uncomment to make it score-increasing
        triggered_rules.append(f"Trial duration seems excessively long (> {max_duration_warn} days)")

    # Rule 8: Impossible Observation Flag
    if record.get('has_impossible_observation', False):
        rule_score += 1; triggered_rules.append("Record contains physiologically impossible observation value")

    # Rule 9: Dosage Out of Range Flag (calculated during prediction)
    if record.get('isDosageOutOfRange_overall', False):
        rule_score += 1; triggered_rules.append("Dosage is outside typical range (1st-99th percentile)")

    return rule_score, triggered_rules

# --- 3. Define Prediction Function ---
def predict_report_anomaly(record_dict):
    """
    Predicts the anomaly score for a single report record using V3 artifacts.
    Assumes record_dict contains features engineered similarly to the training script.
    Calculates dosage normality and drug frequency using loaded artifacts.
    Returns a dictionary with prediction results or an error structure.
    """
    patient_id = record_dict.get("patient_id", "UNKNOWN") # Get patient ID for logging
    logger.info(f"Processing record for patient: {patient_id}")

    # --- Input Validation (Critical Fields) ---
    critical_features = ['age', 'dosage_numeric', 'drugName', 'doctorNotes']
    missing_critical = [f for f in critical_features if pd.isna(record_dict.get(f))]
    if missing_critical:
        error_msg = f"Missing critical features for patient {patient_id}: {missing_critical}"
        logger.error(error_msg)
        # Return error structure matching successful output format
        return {
            "error": error_msg,
            "rule_score": 0,
            "triggered_rules": [],
            "model_predictions": {},
            "final_combined_prediction": {"label": "Error", "numeric": 0}
        }

    # --- Prepare DataFrame from Input ---
    try:
        # Create a DataFrame directly from the input dictionary
        input_df = pd.DataFrame([record_dict]) # Create DataFrame with one row
    except Exception as df_err:
        error_msg = f"Failed to create DataFrame for patient {patient_id}: {df_err}"
        logger.error(error_msg, exc_info=True)
        return { "error": error_msg, "rule_score": 0, "triggered_rules": [], "model_predictions": {}, "final_combined_prediction": {"label": "Error", "numeric": 0} }


    # --- Feature Engineering (Prediction Time) ---
    logger.debug(f"Starting feature engineering for patient {patient_id}")
    # Ensure correct types before calculations
    # Use .get() on the original dict for safety before assigning to DataFrame
    input_df['dosage_numeric'] = pd.to_numeric(record_dict.get('dosage_numeric'), errors='coerce')
    input_df['drugName'] = str(record_dict.get('drugName', 'Unknown')) # Ensure drugName exists as string
    input_df['doctorNotes'] = str(record_dict.get('doctorNotes', '')) # Ensure doctorNotes exists as string

    try:
        # 1. Sentiment Score
        input_df['sentiment_score'] = get_sentiment_score(input_df['doctorNotes'].iloc[0])
        logger.debug(f"  Calculated sentiment score: {input_df['sentiment_score'].iloc[0]:.4f}")

        # 2. Dosage per Age
        # Ensure age and dosage are numeric before division
        input_df['age'] = pd.to_numeric(input_df['age'], errors='coerce')
        input_df['dosage_numeric'] = pd.to_numeric(input_df['dosage_numeric'], errors='coerce')

        # Handle potential division by zero or NaN age
        safe_age = input_df['age'].replace(0, np.nan)
        if pd.notna(input_df['dosage_numeric'].iloc[0]) and pd.notna(safe_age.iloc[0]):
             input_df['dosage_per_age'] = input_df['dosage_numeric'] / safe_age
        else:
             input_df['dosage_per_age'] = np.nan # Keep as NaN initially

        # Fill NaN dosage_per_age with 0 *after* calculation
        input_df['dosage_per_age'] = input_df['dosage_per_age'].fillna(0)
        logger.debug(f"  Calculated dosage_per_age: {input_df['dosage_per_age'].iloc[0]:.4f}")

    except Exception as feat_eng_err:
        error_msg = f"Error during basic feature engineering (sentiment/dosage_per_age) for patient {patient_id}: {feat_eng_err}"
        logger.error(error_msg, exc_info=True)
        return { "error": error_msg, "rule_score": 0, "triggered_rules": [], "model_predictions": {}, "final_combined_prediction": {"label": "Error", "numeric": 0} }


    # 3. Dosage Normality (using loaded stats)
    if dosage_stats and 'dosage_numeric' in input_df.columns:
        mean = dosage_stats.get('overall_mean', 0)
        std = dosage_stats.get('overall_std', 1)
        p01 = dosage_stats.get('overall_p01', 0)
        p99 = dosage_stats.get('overall_p99', 1)
        if std == 0: std = 1

        input_df['dosage_numeric'] = input_df['dosage_numeric'].fillna(mean) # Impute NaN dosage
        # Impute NaN dosage *before* calculating Z-score and range check
        input_df['dosage_numeric'] = input_df['dosage_numeric'].fillna(mean)
        input_df['dosage_zscore_overall'] = (input_df['dosage_numeric'] - mean) / std
        input_df['isDosageOutOfRange_overall'] = ~input_df['dosage_numeric'].between(p01, p99, inclusive='both')
        logger.debug("  Calculated dosage normality features using loaded stats.")
    else:
        input_df['dosage_zscore_overall'] = 0
        input_df['isDosageOutOfRange_overall'] = False
        logger.warning("  Dosage stats missing in artifacts or 'dosage_numeric' column absent. Using default normality features.")

    # 4. Drug Frequency (using loaded map)
    if drug_freq_map and 'drugName' in input_df.columns:
        # Ensure drugName is string, handle potential NaN
        input_df['drugName'] = input_df['drugName'].fillna('Unknown').astype(str)
        # Use .get on the map for unknown drugs
        input_df['drugName_freq'] = input_df['drugName'].apply(lambda x: drug_freq_map.get(x, 0.0))
        logger.debug("  Calculated drug frequency using loaded map.")
    else:
        input_df['drugName_freq'] = 0.0
        logger.warning("  Drug frequency map missing in artifacts or 'drugName' column absent. Using 0.0.")

    # Ensure list columns are lists and exist
    for col in multilabel_features:
        input_value = record_dict.get(col, []) # Get value from original dict
        input_df[col] = [parse_list_string(input_value)] # Apply parsing and assign as list within list

    # Add *all* potentially expected columns (numerical, categorical, text) with defaults
    # This ensures the DataFrame structure matches what the ColumnTransformer expects,
    # even if the input dictionary is sparse.
    all_expected_cols = set(numerical_features + categorical_features + [text_feature] + multilabel_features)
    missing_cols_added = []
    for col in all_expected_cols:
        if col not in input_df.columns:
            missing_cols_added.append(col)
            default_value = None
            # Assign more meaningful defaults where possible
            if col in numerical_features: default_value = 0.0 # Use float for numerical
            elif col in categorical_features: default_value = 'Unknown'
            elif col == text_feature: default_value = ''
            elif col in multilabel_features: default_value = [] # Use empty list
            else: default_value = None # Should not happen if lists are correct

            # Handle specific defaults if needed (overrides generic ones)
            if col == 'age': default_value = np.nan # Let imputation handle age if truly missing
            elif col == 'trialDuration': default_value = 30.0
            elif col == 'symptomImprovementScore': default_value = 5.0
            elif col == 'sideEffectSeverity': default_value = 'None'
            elif col == 'overallHealthStatus': default_value = 'Same'
            elif col == 'gender': default_value = 'Other'
            elif col == 'has_impossible_observation': default_value = False
            elif col == 'dosage_numeric': default_value = np.nan # Let imputation handle dosage

            # Assign the default value
            if col in multilabel_features:
                 input_df[col] = [default_value] # Needs to be list of lists for df assignment
            else:
                 input_df[col] = default_value

    if missing_cols_added:
        logger.warning(f"Patient {patient_id}: Added default values for missing input features: {sorted(missing_cols_added)}")

    # Apply Rule Checks (after calculating derived features)
    # Convert Series back to dict for check_rules
    record_for_rules = input_df.iloc[0].to_dict()
    # Ensure list columns are simple lists for rule checking
    for col in multilabel_features:
        if col in record_for_rules and isinstance(record_for_rules[col], list) and len(record_for_rules[col]) == 1 and isinstance(record_for_rules[col][0], list):
             record_for_rules[col] = record_for_rules[col][0] # Extract inner list
        elif col in record_for_rules and not isinstance(record_for_rules[col], list):
             record_for_rules[col] = parse_list_string(record_for_rules[col]) # Ensure it's a list

    rule_score, triggered_rules = check_rules(record_for_rules, RULE_THRESHOLDS) # Pass thresholds
    logger.info(f"Patient {patient_id}: Rule Check Score: {rule_score}, Triggered: {triggered_rules}")

    # --- Preprocessing using loaded artifacts ---
    logger.debug(f"Starting preprocessing for patient {patient_id}")
    # Separate MLB processing
    mlb_processed_arrays = []
    for col in multilabel_features:
        if col in mlb_transformers:
            mlb = mlb_transformers[col]
            mlb_classes_for_col = mlb_classes.get(col, [])
            if not mlb_classes_for_col: continue
            try:
                data_to_transform = input_df[col].tolist()
                mlb.classes_ = mlb_classes_for_col
                # Filter input data to only include known classes before transform
                data_filtered = [[item for item in sublist if item in mlb.classes_] for sublist in data_to_transform]
                # Ensure data_to_transform contains lists, not potentially other types
                data_to_transform_cleaned = [lst if isinstance(lst, list) else [] for lst in data_to_transform]
                # Filter input data to only include known classes before transform
                data_filtered = [[item for item in sublist if item in mlb.classes_] for sublist in data_to_transform_cleaned]
                transformed = mlb.transform(data_filtered)
                mlb_processed_arrays.append(transformed)
                logger.debug(f"  Applied MLB for {col}")
            except Exception as e:
                 logger.error(f"  Error applying MLB for {col} for patient {patient_id}: {e}", exc_info=True)
                 # Add zeros array matching expected shape
                 num_classes = len(mlb_classes_for_col) if mlb_classes_for_col else 0
                 mlb_processed_arrays.append(np.zeros((len(input_df), num_classes)))
        else:
             # Handle case where MLB transformer wasn't found for a feature in the list
             logger.warning(f"  MLB transformer for feature '{col}' not found in artifacts. Adding zeros.")
             num_classes = len(mlb_classes.get(col, []))
             mlb_processed_arrays.append(np.zeros((len(input_df), num_classes)))

    # Prepare DataFrame for ColumnTransformer - select only expected features
    if not ct_expected_features:
         error_msg = f"Cannot determine expected features for ColumnTransformer for patient {patient_id}."
         logger.error(error_msg)
         return {"error": error_msg, "rule_score": rule_score, "triggered_rules": triggered_rules, "model_predictions": {}, "final_combined_prediction": {"label": "Error", "numeric": 0}}

    # Ensure all columns expected by CT exist in the input_df, using defaults if necessary
    # (This step might be redundant if the previous default-adding step was comprehensive)
    missing_ct_cols_final_check = []
    for feature in ct_expected_features:
        if feature not in input_df.columns:
            missing_ct_cols_final_check.append(feature)
            # Add appropriate default (should ideally not happen if previous step worked)
            if feature in numerical_features: input_df[feature] = 0.0
            elif feature in categorical_features: input_df[feature] = 'Unknown'
            elif feature == text_feature: input_df[feature] = ''
            else: input_df[feature] = None # Fallback
    if missing_ct_cols_final_check:
         logger.warning(f"Patient {patient_id}: Added placeholder columns during final CT check: {missing_ct_cols_final_check}")

    # Prepare DataFrame for ColumnTransformer: Select the original input columns it expects.
    # These are defined in the numerical_features, categorical_features, and text_feature lists.
    ct_input_cols = numerical_features + categorical_features + [text_feature]

    # Ensure all these input columns exist in input_df (redundant check, should be guaranteed by default handling)
    missing_ct_input_cols = [col for col in ct_input_cols if col not in input_df.columns]
    if missing_ct_input_cols:
        # This should ideally not happen due to the earlier default value population.
        error_msg = f"Internal error: Critical input columns for CT missing after default handling: {missing_ct_input_cols}"
        logger.error(error_msg)
        # Return error structure
        return {"error": error_msg, "rule_score": rule_score, "triggered_rules": triggered_rules, "model_predictions": {}, "final_combined_prediction": {"label": "Error", "numeric": 0}}

    # Select the correct input columns in the order expected (assuming the lists define the order)
    input_df_for_ct = input_df[ct_input_cols]
    logger.debug(f"Prepared DataFrame for ColumnTransformer with columns: {list(input_df_for_ct.columns)}")

    # Apply ColumnTransformer
    try:
        # Explicitly clean the text feature column right before transformation
        if text_feature and text_feature in input_df_for_ct.columns:
            cleaned_notes = []
            for note in input_df_for_ct[text_feature]:
                if isinstance(note, str):
                    cleaned_notes.append(note)
                else: # Handles None, NaN, and any other non-string types
                    # Ensure text feature is string type before cleaning
                    input_df_for_ct[text_feature] = input_df_for_ct[text_feature].astype(str)
            cleaned_notes = []
            for note in input_df_for_ct[text_feature]:
                # Basic cleaning (can be expanded)
                cleaned_notes.append(note.strip())
            input_df_for_ct[text_feature] = cleaned_notes
            logger.debug(f"  Explicitly cleaned '{text_feature}' before CT.")

        X_input_ct = ct.transform(input_df_for_ct)
        # Convert sparse matrix to dense if necessary
        if hasattr(X_input_ct, "toarray"): X_input_ct = X_input_ct.toarray()
        logger.debug(f"  ColumnTransformer successful. Shape: {X_input_ct.shape}")
    except Exception as e:
        error_msg = f"Error during ColumnTransformer transformation for patient {patient_id}: {e}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg, "rule_score": rule_score, "triggered_rules": triggered_rules, "model_predictions": {}, "final_combined_prediction": {"label": "Error", "numeric": 0}}

    # Combine CT features and MLB features
    try:
        if X_input_ct.shape[0] == 0 and any(arr.shape[0] > 0 for arr in mlb_processed_arrays):
             X_input_ct = np.zeros((mlb_processed_arrays[0].shape[0], X_input_ct.shape[1]))
        elif any(arr.shape[0] != X_input_ct.shape[0] for arr in mlb_processed_arrays):
             raise ValueError("Mismatch in number of rows between CT and MLB outputs.")

        # Ensure all parts have the same number of rows (should be 1)
        if not all(arr.shape[0] == X_input_ct.shape[0] for arr in mlb_processed_arrays):
             raise ValueError("Mismatch in number of rows between CT and MLB outputs during hstack.")

        X_input_processed = np.hstack([X_input_ct] + mlb_processed_arrays)
        logger.info(f"Patient {patient_id}: Combined all features. Final shape: {X_input_processed.shape}")
    except ValueError as e:
         error_msg = f"Error combining features for patient {patient_id}: {e}"
         logger.error(error_msg, exc_info=True)
         logger.error(f"    CT shape: {X_input_ct.shape}")
         for i, arr in enumerate(mlb_processed_arrays): logger.error(f"    MLB {i} shape: {arr.shape}")
         return {"error": error_msg, "rule_score": rule_score, "triggered_rules": triggered_rules, "model_predictions": {}, "final_combined_prediction": {"label": "Error", "numeric": 0}}

    # --- Get predictions from all models ---
    predictions = {}
    for name, model in models.items():
        try:
            score = np.nan
            label = "Error"
            pred = model.predict(X_input_processed)[0]
            label = "Outlier/Anomaly" if pred == -1 else "Inlier/Normal"

            if hasattr(model, "decision_function"):
                score = model.decision_function(X_input_processed)[0]
            elif hasattr(model, "score_samples"):
                score = model.score_samples(X_input_processed)[0]

            # Ensure score is a standard float, handle potential numpy types
            score = float(score) if pd.notna(score) else None # Use None for NaN scores

            predictions[name] = {"raw_score": score, "prediction_label": label}
            logger.debug(f"  {name} Score: {score}, Prediction: {label}")
        except Exception as e:
            logger.error(f"  Error getting prediction for {name} for patient {patient_id}: {e}", exc_info=True)
            predictions[name] = {"error": str(e)}

    # --- Get Autoencoder Prediction ---
    if autoencoder and autoencoder_threshold is not None:
        try:
            reconstructions = autoencoder.predict(X_input_processed, verbose=0)
            mse = np.mean(np.power(X_input_processed - reconstructions, 2), axis=1)
            ae_pred = -1 if mse[0] > autoencoder_threshold else 1
            ae_mse = float(mse[0]) # Ensure float
            ae_label = "Outlier/Anomaly" if ae_pred == -1 else "Inlier/Normal"
            predictions['Autoencoder'] = {"raw_score": ae_mse, "prediction_label": ae_label}
            logger.debug(f"  Autoencoder MSE: {ae_mse:.6f}, Threshold: {autoencoder_threshold:.6f}, Prediction: {ae_label}")
        except Exception as e:
            logger.error(f"  Error getting prediction for Autoencoder for patient {patient_id}: {e}", exc_info=True)
            predictions['Autoencoder'] = {"error": str(e)}
    else:
        logger.debug("  Skipping Autoencoder prediction (model or threshold not loaded/found).")


    # --- Final Combined Decision ---
    final_prediction_label = "Error"
    final_prediction_numeric = 0 # Default to error/unknown

    # Prioritize rule violations
    if rule_score > 0:
        final_prediction_label = "Outlier/Anomaly"
        final_prediction_numeric = -1
    else:
        # Implement Majority Voting (including Autoencoder if available)
        anomaly_votes = 0
        valid_model_predictions = 0
        # Consider all models present in the predictions dict, excluding those with errors
        model_names_to_vote = [name for name, pred in predictions.items() if 'error' not in pred]

        for name in model_names_to_vote:
            valid_model_predictions += 1
            if predictions[name].get('prediction_label') == 'Outlier/Anomaly':
                anomaly_votes += 1

        # Majority requires > half the *valid* votes
        if valid_model_predictions > 0:
            if anomaly_votes > (valid_model_predictions / 2.0):
                final_prediction_label = "Outlier/Anomaly"
                final_prediction_numeric = -1
                logger.debug(f"  Majority Vote Result: Anomaly ({anomaly_votes}/{valid_model_predictions} votes)")
            else: # If not majority anomaly, predict normal
                final_prediction_label = "Inlier/Normal"
                final_prediction_numeric = 1
                logger.debug(f"  Majority Vote Result: Normal ({anomaly_votes}/{valid_model_predictions} votes)")
        else:
            # Handle case where all models failed but rules passed
            logger.warning(f"Patient {patient_id}: Rules passed but no valid model predictions available. Defaulting final prediction to Normal.")
            final_prediction_label = "Inlier/Normal" # Or maybe 'Undetermined'?
            final_prediction_numeric = 1 # Or 0?

    logger.info(f"Patient {patient_id}: Final Prediction: {final_prediction_label} (Numeric: {final_prediction_numeric})")


    # Return combined results
    return {
        "rule_score": rule_score,
        "triggered_rules": triggered_rules,
        "model_predictions": predictions,
        "final_combined_prediction": {
             "label": final_prediction_label,
             "numeric": final_prediction_numeric
        }
    }
