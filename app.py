import pandas as pd
import numpy as np
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import json, re
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from transformers import pipeline
import requests
from google import genai
import ast
import os
from dotenv import load_dotenv
load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

specs = pd.read_csv('./MetalliSense.metal_grade_specs.csv')
readings = pd.read_csv('./MetalliSense.spectrometer_readings.csv')

range_cols = [col for col in specs.columns if col.startswith('composition_range.')]
merged = readings.merge(specs, on='metal_grade', how='left')
for col in set(c.split("[")[0] for c in range_cols):
    low = specs[col + "[0]"]
    high = specs[col + "[1]"]
    elem = col.replace("composition_range.", "")
    reading_col = f"composition.{elem}"
    
    if reading_col in merged.columns:
        merged[f"delta_{elem}"] = merged[reading_col].apply(
            lambda x: 0 if pd.isna(x) else
            (low.mean() - x if x < low.mean() else
             (x - high.mean() if x > high.mean() else 0))
        )

feature_cols = [c for c in readings.columns if c.startswith('composition.')]
target_cols = [c for c in merged.columns if c.startswith('delta_')]

grade_encoded = pd.get_dummies(merged["metal_grade"], prefix="grade")
X = pd.concat([merged[feature_cols], grade_encoded], axis=1).fillna(0)
Y = merged[target_cols].fillna(0)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = MultiOutputRegressor(
    xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        learning_rate=0.1,
    )
)
model.fit(X_train_scaled, Y_train)


app = FastAPI(title="MetalliSense API")

class PredictRequest(BaseModel):
    metal_grade: str
    composition: List[float]  
    
class PredictResponse(BaseModel):
    metal_grade: str
    adjustments: dict
    recommendations: dict
    
def safe_json_parse(text: str) -> dict:
    try:
        return json.loads(text)
    except:
        try:
            text = text.replace("'", '"')
            return json.loads(text)
        except:
            try:
                return ast.literal_eval(text)
            except:
                return {}
    


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


def get_recommendations(metal_grade: str, adjustments: Dict[str,str]) -> Dict[str,str]:
    recommendations = {}
    
    
    for elem, action in adjustments.items():
        if action == "ok":
            continue
        if "add" in action:
            recommendations[elem] = ELEMENT_FALLBACK.get(elem, {}).get("add", "No recommendation")
        elif "reduce" in action:
            recommendations[elem] = ELEMENT_FALLBACK.get(elem, {}).get("reduce", "No recommendation")

    # Prepare prompt for Gemini
    prompt = f"""
    You are a metallurgical expert.
    Metal grade: {metal_grade}
    Adjustments: {adjustments}

    Rules:
    - If an element must be reduced, suggest which material/input should be added or avoided.
    - If it must be increased, suggest what alloying element or scrap to add.
    - If it's 'ok', skip.
    - Answer in strict JSON: {{ "element": "recommendation", ... }}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            # temperature=0,
            # max_output_tokens=300
        )
        text_output = response.text
        json_str = re.search(r"\{.*\}", text_output, re.S).group()
        recommendations.update(json.loads(json_str))
    except Exception as e:
        print("Gemini API error:", e)
        pass

    return recommendations


@app.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    if data.metal_grade not in specs["metal_grade"].values:
        raise HTTPException(status_code=404, detail="Metal grade not found")

    row = {f"composition.{i}": val for i, val in enumerate(data.composition)}
    df = pd.DataFrame([row])

    for g in specs["metal_grade"].unique():
        df[f"grade_{g}"] = 1 if data.metal_grade == g else 0
    X_scaled = scaler.transform(df.reindex(columns=X.columns, fill_value=0))

    pred = model.predict(X_scaled)[0]

    adjustments = {}
    for col, adj in zip(target_cols, pred):
        elem = col.replace("delta_", "")
        if adj > 0:
            action = f"reduce {adj:.3f}%"
        elif adj < 0:
            action = f"add {abs(adj):.3f}%"
        else:
            action = "ok"
        adjustments[elem] = action
    
    recommendations = get_recommendations(data.metal_grade,adjustments)
    print("Adjustments:", adjustments)
    print(f"Recommendation: \n {recommendations}")
    for elem in adjustments:
        if elem not in ELEMENT_FALLBACK:
            print(f"No fallback found for element: {elem}")

    
    return PredictResponse(
        metal_grade=data.metal_grade,
        adjustments=adjustments,
        recommendations = recommendations
    )
