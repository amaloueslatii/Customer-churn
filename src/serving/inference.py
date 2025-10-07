

import os
import pandas as pd
import mlflow
from pathlib import Path
import glob

from pathlib import Path
import os
import glob
import mlflow
import pandas as pd

# Allow override via env, default to /app/model (Docker) or fallback to mlruns below
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/model"))

# ---- Load model (primary + fallback) ----
try:
    model = mlflow.pyfunc.load_model(str(MODEL_DIR))  # Path -> str for mlflow
    print(f"✅ Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    print(f"❌ Failed to load model from {MODEL_DIR}: {e}")
    # Fallback for local development: pick the latest mlruns/*/*/artifacts/model
    try:
        # Use Path/glob for cross-platform
        candidates = list(Path("./mlruns").glob("*/*/artifacts/model"))
        if not candidates:
            raise FileNotFoundError("No model found under ./mlruns/*/*/artifacts/model")
        latest_model = max(candidates, key=lambda p: p.stat().st_mtime)
        model = mlflow.pyfunc.load_model(str(latest_model))
        MODEL_DIR = latest_model  # point MODEL_DIR to the actual loaded path
        print(f"✅ Fallback: Loaded model from {latest_model}")
    except Exception as fallback_error:
        raise Exception(f"Failed to load model: {e}. Fallback failed: {fallback_error}")

# === FEATURE SCHEMA LOADING ===
# Try feature_columns.txt in both places:
#  1) artifacts/model/feature_columns.txt
#  2) artifacts/feature_columns.txt  (run-root artifact)
try:
    model_dir = Path(MODEL_DIR)
    # If a file path were ever passed, normalize to its folder
    if model_dir.is_file():
        model_dir = model_dir.parent

    candidates = [
        model_dir / "feature_columns.txt",
        model_dir.parent / "feature_columns.txt",  # parent == artifacts/
    ]

    feature_file = next((p for p in candidates if p.exists()), None)
    if feature_file is None:
        tried = [str(p) for p in candidates]
        raise FileNotFoundError(f"feature_columns.txt not found. Tried: {tried}")

    with feature_file.open("r", encoding="utf-8") as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]

    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from {feature_file}")

except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}")


# === FEATURE TRANSFORMATION CONSTANTS ===


# Deterministic binary feature mappings (consistent with training)
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},           # Demographics
    "Partner": {"No": 0, "Yes": 1},               # Has partner
    "Dependents": {"No": 0, "Yes": 1},            # Has dependents  
    "PhoneService": {"No": 0, "Yes": 1},          # Phone service
    "PaperlessBilling": {"No": 0, "Yes": 1},      # Billing preference
}

# Numeric columns that need type coercion
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply identical feature transformations as used during model training.
    
   
    """
    df = df.copy()
    
    # Clean column names (remove any whitespace)
    df.columns = df.columns.str.strip()
    
    # === STEP 1: Numeric Type Coercion ===
    # Ensure numeric columns are properly typed (handle string inputs)
    for c in NUMERIC_COLS:
        if c in df.columns:
            # Convert to numeric, replacing invalid values with NaN
            df[c] = pd.to_numeric(df[c], errors="coerce")
            # Fill NaN with 0 (same as training preprocessing)
            df[c] = df[c].fillna(0)
    
    # === STEP 2: Binary Feature Encoding ===
    # Apply deterministic mappings for binary features
    # CRITICAL: Must use exact same mappings as training
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)                    # Convert to string
                .str.strip()                    # Remove whitespace
                .map(mapping)                   # Apply binary mapping
                .astype("Int64")                # Handle NaN values
                .fillna(0)                      # Fill unknown values with 0
                .astype(int)                    # Final integer conversion
            )
    
    # === STEP 3: One-Hot Encoding for Remaining Categorical Features ===
    # Find remaining object/categorical columns (not in BINARY_MAP)
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        # Apply one-hot encoding with drop_first=True (same as training)
        # This prevents multicollinearity by dropping the first category
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    
    # === STEP 4: Boolean to Integer Conversion ===
    # Convert any boolean columns to integers (XGBoost compatibility)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
    
    # === STEP 5: Feature Alignment with Training Schema ===
   
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    
    return df

def predict(input_dict: dict) -> str:
    """
    Predict customer churn based on input features.
    """
    
    # === STEP 1: Convert Input to DataFrame ===
    # Create single-row DataFrame for pandas transformations
    df = pd.DataFrame([input_dict])
    
    # === STEP 2: Apply Feature Transformations ===
    # Use the same transformation pipeline as training
    df_enc = _serve_transform(df)
    
    # === STEP 3: Generate Model Prediction ===
    # Call the loaded MLflow model for inference
    # The model returns predictions in various formats depending on the ML library
    try:
        preds = model.predict(df_enc)
        
        # Normalize prediction output to consistent format
        if hasattr(preds, "tolist"):
            preds = preds.tolist()  # Convert numpy array to list
            
        # Extract single prediction value (for single-row input)
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds
            
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")
    
    # === STEP 4: Convert to Business-Friendly Output ===
    # Convert binary prediction (0/1) to actionable business language
    if result == 1:
        return "Likely to churn"      # High risk - needs intervention
    else:
        return "Not likely to churn"  # Low risk - maintain normal service
