import os
import re
import json
import ast
from typing import List, Dict, Union

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb

from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI(title="MetalliSense API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Load specs & readings ----------
specs = pd.read_csv('./MetalliSense.metal_grade_specs.csv')
readings = pd.read_csv('./MetalliSense.spectrometer_readings.csv')

comp_range_cols = [c for c in specs.columns if c.startswith('composition_range.')]
range_bases = sorted(set(c.split("[")[0] for c in comp_range_cols))
feature_cols = [c for c in readings.columns if c.startswith('composition.')]

merged = readings.merge(specs, on='metal_grade', how='left')

# ---------- Compute per-element deltas ----------
IMPURITIES = ["P", "S"]
for base in range_bases:
    elem = base.replace("composition_range.", "")
    reading_col = f"composition.{elem}"
    low_col = f"{base}[0]"
    high_col = f"{base}[1]"

    if reading_col not in merged.columns or low_col not in merged.columns or high_col not in merged.columns:
        merged[f"delta_{elem}"] = 0.0
        continue

    x = pd.to_numeric(merged[reading_col], errors='coerce')
    low = pd.to_numeric(merged[low_col], errors='coerce')
    high = pd.to_numeric(merged[high_col], errors='coerce')

    if elem in IMPURITIES:
        merged[f"delta_{elem}"] = np.where(x > high, x - high, 0.0)
    else:
        merged[f"delta_{elem}"] = np.where(x < low, x - low, np.where(x > high, x - high, 0.0))

target_cols = [c for c in merged.columns if c.startswith('delta_')]

# ---------- Prepare training data ----------
grade_encoded = pd.get_dummies(merged["metal_grade"], prefix="grade")
X = pd.concat([merged[feature_cols], grade_encoded], axis=1).fillna(0)
Y = merged[target_cols].fillna(0)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = MultiOutputRegressor(
    xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200, learning_rate=0.1)
)
model.fit(X_train_scaled, Y_train)

# ---------- API Models ----------
class PredictRequest(BaseModel):
    metal_grade: str
    composition: Union[List[float], Dict[str, float]]

class PredictResponse(BaseModel):
    metal_grade: str
    adjustments: Dict[str, str]
    recommendations: Dict[str, str]

ELEMENT_FALLBACK = {
    "C": {"add": "Add carburizer to increase C.", "reduce": "Dilute with pig iron to reduce C."},
    "Si": {"add": "Add ferrosilicon.", "reduce": "Use low-silicon scrap."},
    "Mn": {"add": "Add ferromanganese.", "reduce": "Dilute with low-Mn scrap."},
    "P": {"add": "", "reduce": "Use low-phosphorus raw materials."},
    "S": {"add": "", "reduce": "Add desulfurization agents like CaO or Mg."},
    "Cr": {"add": "Add ferrochrome.", "reduce": "Dilute with non-chromium scrap."},
    "Ni": {"add": "Add ferronickel or nickel scrap.", "reduce": "Use less Ni-rich input."},
    "Mo": {"add": "Add ferromolybdenum.", "reduce": "Dilute with Mo-free scrap."},
    "Cu": {"add": "Add copper scrap or alloys.", "reduce": "Avoid copper-rich materials."},
    "Fe": {"add": "Add iron scrap.", "reduce": "Dilute with low-iron scrap."},
    "V": {"add": "Add ferrovanadium.", "reduce": "Avoid vanadium-rich scrap."},
    "Nb": {"add": "Add niobium ferroalloy.", "reduce": "Avoid niobium-rich input."},
    "Mg": {"add": "Add magnesium.", "reduce": "Avoid magnesium-rich materials."},
    "Ti": {"add": "Add titanium scrap.", "reduce": "Avoid titanium-rich input."}
}

def safe_json_parse(text: str) -> dict:
    try:
        return json.loads(text)
    except:
        try:
            return json.loads(text.replace("'", '"'))
        except:
            try:
                return ast.literal_eval(text)
            except:
                return {}

def compute_adjustments_from_spec(metal_grade: str, composition_map: Dict[str, float]) -> Dict[str, float]:
    spec_row = specs[specs["metal_grade"] == metal_grade]
    if spec_row.empty:
        raise ValueError("metal grade not found in specs")
    spec = spec_row.iloc[0]
    adjustments = {}
    for col in feature_cols:
        elem = col.split(".", 1)[1]
        val = composition_map.get(elem, None)
        if val is None:
            adjustments[elem] = 0.0
            continue
        low = float(spec.get(f"composition_range.{elem}[0]", np.nan))
        high = float(spec.get(f"composition_range.{elem}[1]", np.nan))
        if elem in IMPURITIES:
            adjustments[elem] = val - high if val > high else 0.0
        else:
            if val < low:
                adjustments[elem] = val - low
            elif val > high:
                adjustments[elem] = val - high
            else:
                adjustments[elem] = 0.0
    return adjustments

def get_recommendations(metal_grade: str, adjustments: Dict[str,str]) -> Dict[str,str]:
    recs = {}
    for elem, action in adjustments.items():
        if action == "ok":
            continue
        if "add" in action:
            recs[elem] = ELEMENT_FALLBACK.get(elem, {}).get("add", "No recommendation")
        elif "reduce" in action:
            recs[elem] = ELEMENT_FALLBACK.get(elem, {}).get("reduce", "No recommendation")
    # Ask LLM for additional precise recommendations, but guard and sanitize the output.
    try:
        prompt = f"""
            You are a metallurgical expert. Provide precise alloy addition recommendations for a given metal grade. Do not include explanations about raw materials, processes, or anything else. Only give clear alloy recommendations in JSON format.

            Metal grade: {metal_grade}  
            Adjustments: {adjustments}  

            Example recommendation format:  
            {{"Mg": "Add ferromagnesium to increase Mg."}}
            when giving recommendations instead of giving the same metal name in the recommendation, try to give alloy names. Tell what has to be done instead of telling what could have been done.
            Answer strictly with a single JSON object mapping element symbols to recommendation strings (no wrapper keys).
            """

        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        raw_text = response.text if getattr(response, 'text', None) else ''
        # extract first JSON-like object from the text
        m = re.search(r"\{.*\}", raw_text, re.S)
        parsed = safe_json_parse(m.group() if m else raw_text)
        # sanitize: only accept dict[str, str] and merge into recs
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                # skip any nested structures or wrapper keys
                if k == 'recommendations':
                    # if user model returned a nested recommendations dict, flatten it
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            if isinstance(kk, str) and isinstance(vv, str):
                                recs[kk] = vv
                    continue
                if isinstance(k, str) and isinstance(v, str):
                    recs[k] = v
    except Exception as e:
        # Don't fail the API if the LLM call fails; just return fallback recs
        print("Gemini API error:", e)
    # final sanitize: ensure all values are strings
    sanitized = {}
    for k, v in recs.items():
        try:
            sanitized[str(k)] = str(v)
        except Exception:
            sanitized[str(k)] = ""
    return sanitized
    return recs

@app.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    if data.metal_grade not in specs["metal_grade"].values:
        raise HTTPException(status_code=404, detail="Metal grade not found")

    # normalize composition input
    if isinstance(data.composition, dict):
        comp_map = {k: float(v) for k, v in data.composition.items()}
    elif isinstance(data.composition, list):
        if len(data.composition) != len(feature_cols):
            raise HTTPException(status_code=400, detail="composition length mismatch")
        comp_map = {col.split(".",1)[1]: float(val) for col, val in zip(feature_cols, data.composition)}
    else:
        raise HTTPException(status_code=400, detail="composition must be list or dict")

    # model-based adjustments
    row = {col: 0.0 for col in X.columns}
    for elem, val in comp_map.items():
        col_name = f"composition.{elem}"
        if col_name in row:
            row[col_name] = val
    for g in specs["metal_grade"].unique():
        row[f"grade_{g}"] = 1.0 if data.metal_grade == g else 0.0

    df = pd.DataFrame([row])
    X_scaled = scaler.transform(df.reindex(columns=X.columns, fill_value=0))
    pred = model.predict(X_scaled)[0]

    adjustments = {}
    for col, adj in zip(target_cols, pred):
        elem = col.replace("delta_", "")
        if adj > 0:
            adjustments[elem] = f"reduce {adj:.3f}%"
        elif adj < 0:
            adjustments[elem] = f"add {abs(adj):.3f}%"
        else:
            adjustments[elem] = "ok"

    # deterministic adjustments (cross-check)
    try:
        det_adj = compute_adjustments_from_spec(data.metal_grade, comp_map)
        for elem, delta in det_adj.items():
            if delta > 0:
                adjustments[elem] = f"reduce {delta:.3f}%"
            elif delta < 0:
                adjustments[elem] = f"add {abs(delta):.3f}%"
            else:
                adjustments[elem] = "ok"
    except Exception as e:
        print("Deterministic adjustment error:", e)

    recommendations = get_recommendations(data.metal_grade, adjustments)
    print("Adjustments:", adjustments)
    print("Recommendations:", recommendations)
    return PredictResponse(
        metal_grade=data.metal_grade,
        adjustments=adjustments,
        recommendations=recommendations
    )
