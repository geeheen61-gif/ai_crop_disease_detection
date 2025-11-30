import os
import io
import time
import json
import threading
import sqlite3
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import re

try:
    import serial
except Exception:
    serial = None

try:
    import keras
    import h5py
    import zipfile
except Exception:
    keras = None
    h5py = None
    zipfile = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

app = Flask(__name__)
CORS(app)

conn = sqlite3.connect("distance.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS distance_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        distance REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """
)
conn.commit()

ser_port = os.getenv("ESP32_SERIAL_PORT", "COM3")
ser_baud = int(os.getenv("ESP32_SERIAL_BAUD", "9600"))
ser = None
if serial is not None:
    try:
        ser = serial.Serial(ser_port, ser_baud)
    except Exception:
        ser = None

current_distance = 0.0
last_saved_distance = 0.0
sensor_available = ser is not None

def read_from_arduino():
    global current_distance
    while True:
        if ser is None:
            time.sleep(1)
            continue
        try:
            line = ser.readline().decode().strip()
            current_distance = float(line)
        except Exception:
            time.sleep(0.1)

def save_to_db():
    global last_saved_distance
    while True:
        time.sleep(60)
        try:
            cursor.execute("INSERT INTO distance_records(distance) VALUES (?)", (current_distance,))
            conn.commit()
            last_saved_distance = current_distance
        except Exception:
            pass

threading.Thread(target=read_from_arduino, daemon=True).start()
threading.Thread(target=save_to_db, daemon=True).start()

MODEL = None
MODEL_INFO = {}
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]
CLASS_NAMES_SRC = None

CATEGORY_OF = {
    'Apple___Apple_scab': 'Leaf Spot',
    'Apple___Black_rot': 'Blight',
    'Apple___Cedar_apple_rust': 'Rust',
    'Apple___healthy': 'Healthy',
    'Blueberry___healthy': 'Healthy',
    'Cherry_(including_sour)___Powdery_mildew': 'Blight',
    'Cherry_(including_sour)___healthy': 'Healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Leaf Spot',
    'Corn_(maize)___Common_rust_': 'Rust',
    'Corn_(maize)___Northern_Leaf_Blight': 'Blight',
    'Corn_(maize)___healthy': 'Healthy',
    'Grape___Black_rot': 'Blight',
    'Grape___Esca_(Black_Measles)': 'Blight',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Leaf Spot',
    'Grape___healthy': 'Healthy',
    'Orange___Haunglongbing_(Citrus_greening)': 'Nutrient deficiency',
    'Peach___Bacterial_spot': 'Leaf Spot',
    'Peach___healthy': 'Healthy',
    'Pepper,_bell___Bacterial_spot': 'Leaf Spot',
    'Pepper,_bell___healthy': 'Healthy',
    'Potato___Early_blight': 'Blight',
    'Potato___Late_blight': 'Blight',
    'Potato___healthy': 'Healthy',
    'Raspberry___healthy': 'Healthy',
    'Soybean___healthy': 'Healthy',
    'Squash___Powdery_mildew': 'Blight',
    'Strawberry___Leaf_scorch': 'Blight',
    'Strawberry___healthy': 'Healthy',
    'Tomato___Bacterial_spot': 'Leaf Spot',
    'Tomato___Early_blight': 'Blight',
    'Tomato___Late_blight': 'Blight',
    'Tomato___Leaf_Mold': 'Blight',
    'Tomato___Septoria_leaf_spot': 'Leaf Spot',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Blight',
    'Tomato___Target_Spot': 'Leaf Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Blight',
    'Tomato___Tomato_mosaic_virus': 'Blight',
    'Tomato___healthy': 'Healthy'
}

CATEGORY_GUIDE = {
    'Leaf Spot': "Remove infected leaves; improve airflow; avoid overhead watering; apply copper or chlorothalonil per label; sanitize tools; rotate crops.",
    'Blight': "Prune infected tissue; dispose, do not compost; avoid wet foliage; apply approved fungicide; use resistant varieties; rotate crops.",
    'Rust': "Remove rusted leaves; increase spacing; water at soil level; apply sulfur or copper per label; remove alternate hosts.",
    'Nutrient deficiency': "Add balanced NPK; use compost; ensure soil pH ~6–7; avoid overwatering; apply micronutrients if leaf veins stay green.",
    'Healthy': "Maintain spacing; water at roots; mulch; monitor weekly; keep tools clean; balanced fertilization."
}

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def try_load_class_names():
    global CLASS_NAMES
    global CLASS_NAMES_SRC
    paths = [
        os.path.join(os.getcwd(), "disease_class_names.json"),
        os.path.join(os.getcwd(), "class_names.json"),
        os.path.join("Plant_Disease_Detection_Model", "class_names.json"),
        os.getenv("MODEL_CLASS_NAMES_JSON", "").strip(),
    ]
    for p in paths:
        if not p:
            continue
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    names = [str(x) for x in data]
                elif isinstance(data, dict) and isinstance(data.get("class_names"), list):
                    names = [str(x) for x in data["class_names"]]
                else:
                    names = None
                if names:
                    CLASS_NAMES = names
                    CLASS_NAMES_SRC = p
                    return True
            except Exception:
                pass
    return False

def category_for(name):
    n = str(name or "").lower()
    c = CATEGORY_OF.get(name)
    if c:
        return c
    if "healthy" in n:
        return "Healthy"
    if "rust" in n:
        return "Rust"
    if "spot" in n:
        return "Leaf Spot"
    if "mold" in n:
        return "Blight"
    if "blight" in n:
        return "Blight"
    if ("huanglongbing" in n) or ("greening" in n) or ("yellow" in n and "curl" in n):
        return "Nutrient deficiency"
    return "Blight"

try_load_class_names()

def default_gemini_key():
    k = os.getenv("GEMINI_API_KEY")
    if k:
        return k
    try:
        env_path = os.path.join(os.getcwd(), ".env")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY="):
                        return line.split("=", 1)[1].strip()
    except Exception:
        pass
    try:
        p = os.path.join("Plant_Disease_Detection_Model", "main.py")
        with open(p, "r", encoding="utf-8") as f:
            s = f.read()
        m = re.search(r'API_KEY_DEFAULT\s*=\s*"([^"]+)"', s)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None

def load_model():
    global MODEL
    global MODEL_INFO
    if MODEL is not None:
        return MODEL
    if keras is None:
        return None
    if str(os.getenv("LIGHT_MODE", "")).lower() in {"1","true","yes"}:
        return None
    env_h5 = os.getenv("MODEL_H5", "").strip()
    env_keras = os.getenv("MODEL_KERAS", "").strip()
    env_json = os.getenv("MODEL_JSON", "").strip()
    env_weights = os.getenv("MODEL_WEIGHTS_H5", "").strip()
    try:
        if env_json and env_weights and os.path.exists(env_json) and os.path.exists(env_weights):
            try:
                with open(env_json, "r", encoding="utf-8") as f:
                    arch = f.read()
                m = keras.models.model_from_json(arch)
                m.load_weights(env_weights)
                MODEL = m
                MODEL_INFO = {"source": "json+h5", "json": env_json, "weights": env_weights}
                return MODEL
            except Exception:
                pass
        if env_keras and os.path.exists(env_keras):
            try:
                MODEL = keras.models.load_model(env_keras, compile=False, safe_mode=False)
                MODEL_INFO = {"source": "keras", "path": env_keras}
                return MODEL
            except Exception:
                pass
        if env_h5 and os.path.exists(env_h5):
            try:
                MODEL = keras.models.load_model(env_h5, compile=False, safe_mode=False)
                MODEL_INFO = {"source": "h5", "path": env_h5}
                return MODEL
            except Exception:
                pass
    except Exception:
        pass
    candidates = [
        os.path.join("Plant_Disease_Detection_Model", "trained_model.keras"),
        os.path.join("Plant_Disease_Detection_Model", "trained_model.h5"),
        os.path.join(os.getcwd(), "trained_model.keras"),
        os.path.join(os.getcwd(), "trained_model.h5"),
        os.path.join(os.getcwd(), "model.h5"),
        os.path.join(os.getcwd(), "disease_model.keras"),
        os.path.join(os.getcwd(), "disease_model.h5"),
        os.path.join(os.getcwd(), "improved_model.keras"),
        os.path.join(os.getcwd(), "improved_model.h5"),
    ]
    for p in candidates:
        try:
            MODEL = keras.models.load_model(p, compile=False, safe_mode=False)
            MODEL_INFO = {"source": "auto", "path": p}
            return MODEL
        except Exception:
            pass
    try:
        m = keras.Sequential([
            keras.layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(128,128,3), name='conv2d'),
            keras.layers.Conv2D(32, 3, activation='relu', name='conv2d_1'),
            keras.layers.MaxPooling2D(pool_size=2, strides=2, name='max_pooling2d'),
            keras.layers.Conv2D(64, 3, padding='same', activation='relu', name='conv2d_2'),
            keras.layers.Conv2D(64, 3, activation='relu', name='conv2d_3'),
            keras.layers.MaxPooling2D(pool_size=2, strides=2, name='max_pooling2d_1'),
            keras.layers.Conv2D(128, 3, padding='same', activation='relu', name='conv2d_4'),
            keras.layers.Conv2D(128, 3, activation='relu', name='conv2d_5'),
            keras.layers.MaxPooling2D(pool_size=2, strides=2, name='max_pooling2d_2'),
            keras.layers.Conv2D(256, 3, padding='same', activation='relu', name='conv2d_6'),
            keras.layers.Conv2D(256, 3, activation='relu', name='conv2d_7'),
            keras.layers.MaxPooling2D(pool_size=2, strides=2, name='max_pooling2d_3'),
            keras.layers.Conv2D(512, 3, padding='same', activation='relu', name='conv2d_8'),
            keras.layers.Conv2D(512, 3, activation='relu', name='conv2d_9'),
            keras.layers.MaxPooling2D(pool_size=2, strides=2, name='max_pooling2d_4'),
            keras.layers.Dropout(0.25, name='dropout'),
            keras.layers.Flatten(name='flatten'),
            keras.layers.Dense(1500, activation='relu', name='dense'),
            keras.layers.Dropout(0.4, name='dropout_1'),
            keras.layers.Dense(38, activation='softmax', name='dense_1'),
        ])
        kpaths = [
            os.path.join("Plant_Disease_Detection_Model", "trained_model.keras"),
            os.path.join(os.getcwd(), "trained_model.keras"),
        ]
        z = None
        for kpath in kpaths:
            if os.path.exists(kpath):
                try:
                    z = zipfile.ZipFile(kpath)
                    break
                except Exception:
                    z = None
        if z is None:
            raise Exception("no_zipped_keras")
        b = z.read('model.weights.h5')
        f = h5py.File(io.BytesIO(b), 'r')
        layer_group = f['layers']
        for layer in m.layers:
            if layer.name in layer_group:
                vars_group = layer_group[layer.name]['vars']
                keys = sorted([k for k in vars_group.keys()], key=lambda x: int(x))
                arrays = [vars_group[k][...] for k in keys]
                if arrays:
                    try:
                        layer.set_weights(arrays)
                    except Exception:
                        pass
        f.close()
        MODEL = m
        MODEL_INFO = {"source": "zipped_keras_weights"}
        return MODEL
    except Exception:
        return None

def classify_image(file_like):
    image = Image.open(file_like).convert("RGB")
    try:
        model = load_model()
    except Exception:
        model = None
    if model is not None:
        img128 = image.resize((128, 128))
        arr = np.array(img128, dtype=np.float32)
        arr = np.array([arr])
        try:
            prediction = model.predict(arr)
            probs = prediction[0]
            top_idx = int(np.argsort(probs)[::-1][0])
            top3_idx = np.argsort(probs)[::-1][:3]
            top3 = [(CLASS_NAMES[i], float(probs[i])) for i in top3_idx]
            cats = ['Leaf Spot','Blight','Rust','Nutrient deficiency','Healthy']
            cat_probs = {c: 0.0 for c in cats}
            for i, p in enumerate(probs):
                c = category_for(CLASS_NAMES[i])
                if c in cat_probs:
                    cat_probs[c] += float(p)
            cat_items = [(c, cat_probs[c]) for c in cats]
            cat_items.sort(key=lambda x: x[1], reverse=True)
            return top_idx, top3, cat_items[0][0], cat_items[:3]
        except Exception:
            pass
    hsv = image.resize((256, 256)).convert('HSV')
    arr = np.array(hsv, dtype=np.uint8)
    H = arr[:,:,0].astype(np.float32) * (360.0/255.0)
    S = arr[:,:,1].astype(np.float32) / 255.0
    V = arr[:,:,2].astype(np.float32) / 255.0
    green = ((H>=35)&(H<=85)&(S>0.2)).mean()
    yellow = ((H>=20)&(H<=35)&(S>0.25)).mean()
    rusty = (((H>=0)&(H<=20))&(S>0.35)&(V>0.35)).mean()
    dark = (V<0.25).mean()
    cats = ['Leaf Spot','Blight','Rust','Nutrient deficiency','Healthy']
    scores = {
        'Healthy': max(0.0, green - (yellow+rusty+dark)/2),
        'Rust': rusty,
        'Nutrient deficiency': max(0.0, yellow - rusty/2),
        'Leaf Spot': max(0.0, dark - yellow/3),
        'Blight': max(0.0, 1.0 - green - 0.5*yellow - 0.3*rusty),
    }
    items = sorted([(k, float(v)) for k, v in scores.items()], key=lambda x: x[1], reverse=True)
    total = sum(v for _, v in items) or 1.0
    items = [(k, v/total) for k, v in items]
    cat_top = items[0][0]
    cat_top3 = items[:3]
    if cat_top == 'Rust':
        top3 = [('Corn_(maize)__Common_rust', items[0][1]), ('Cherry_(including_sour)__Powdery_mildew', items[1][1]), ('Apple___Cedar_apple_rust', items[2][1])]
    elif cat_top == 'Leaf Spot':
        top3 = [('Tomato___Septoria_leaf_spot', items[0][1]), ('Apple___Apple_scab', items[1][1]), ('Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', items[2][1])]
    elif cat_top == 'Nutrient deficiency':
        top3 = [('Orange___Haunglongbing_(Citrus_greening)', items[0][1]), ('Tomato___Tomato_Yellow_Leaf_Curl_Virus', items[1][1]), ('Potato___healthy', items[2][1])]
    elif cat_top == 'Healthy':
        top3 = [('Tomato___healthy', items[0][1]), ('Grape___healthy', items[1][1]), ('Pepper,_bell___healthy', items[2][1])]
    else:
        top3 = [('Tomato___Early_blight', items[0][1]), ('Potato___Early_blight', items[1][1]), ('Corn_(maize)__Northern_Leaf_Blight', items[2][1])]
    return 0, top3, cat_top, cat_top3

def assess_plant_like(image):
    hsv = image.resize((256, 256)).convert('HSV')
    arr = np.array(hsv, dtype=np.uint8)
    H = arr[:,:,0].astype(np.float32) * (360.0/255.0)
    S = arr[:,:,1].astype(np.float32) / 255.0
    V = arr[:,:,2].astype(np.float32) / 255.0
    green = ((H>=35)&(H<=85)&(S>0.20)&(V>0.20)).mean()
    yellow = ((H>=20)&(H<=35)&(S>0.25)&(V>0.25)).mean()
    brown = (((H>=10)&(H<=30))&(S>0.20)&(V<0.45)).mean()
    sky = (((H>=180)&(H<=260))&(S<0.25)&(V>0.60)).mean()
    plant_score = float(green*0.6 + yellow*0.25 - sky*0.2 - brown*0.15)
    return plant_score

def guidance_text(category, top3_classes, api_key):
    if genai is None or not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        classes_text = ", ".join([f"{n} ({p*100:.1f}%)" for n, p in top3_classes])
        prompt = (
            f"You are an agronomy assistant. The image was predicted as '{category}'. "
            f"Top classes: {classes_text}. Provide a concise summary: 1) disease overview, 2) key symptoms, 3) immediate actions, 4) low-cost treatment plan, 5) prevention. "
            f"Write for small farmers. Use clear steps. Avoid brand names. Keep under 1200 characters."
        )
        resp = model.generate_content(prompt)
        txt = getattr(resp, 'text', None)
        if txt:
            return txt
        if getattr(resp, 'candidates', None):
            try:
                return resp.candidates[0].content.parts[0].text
            except Exception:
                return None
        return None
    except Exception:
        return None

@app.get("/health")
def health():
    return jsonify({"ok": True, "sensor_available": sensor_available})

@app.get("/status")
def status():
    return jsonify({
        "keras_available": bool(keras),
        "model_loaded": MODEL is not None,
        "light_mode": str(os.getenv("LIGHT_MODE", "")).lower() in {"1","true","yes"},
        "has_trained_model_keras": os.path.exists(os.path.join("Plant_Disease_Detection_Model", "trained_model.keras")),
        "has_trained_model_h5": os.path.exists(os.path.join("Plant_Disease_Detection_Model", "trained_model.h5")),
        "model_info": MODEL_INFO,
        "class_names_source": CLASS_NAMES_SRC,
        "num_classes": len(CLASS_NAMES),
        "available_models": [p for p in [
            os.path.join(os.getcwd(), "model.h5"),
            os.path.join(os.getcwd(), "model.keras"),
            os.path.join(os.getcwd(), "model.json"),
            os.path.join(os.getcwd(), "disease_model.h5"),
            os.path.join(os.getcwd(), "disease_model.keras"),
            os.path.join(os.getcwd(), "disease_class_names.json"),
            os.path.join(os.getcwd(), "improved_model.h5"),
            os.path.join(os.getcwd(), "improved_model.keras"),
            os.path.join(os.getcwd(), "improved_model.json"),
            os.path.join("Plant_Disease_Detection_Model", "trained_model.h5"),
            os.path.join("Plant_Disease_Detection_Model", "trained_model.keras"),
        ] if os.path.exists(p)]
    })

@app.get("/distance")
def get_distance():
    return jsonify({"distance": current_distance, "sensor_available": sensor_available})

@app.get("/last-saved")
def get_last_saved():
    return jsonify({"saved_distance": last_saved_distance})

@app.post("/predict")
def predict():
    try:
        api_key = request.form.get("gemini_api_key") or (request.json.get("gemini_api_key") if request.is_json else None) or default_gemini_key()
        include_guidance = (request.form.get("include_guidance") or (request.json.get("include_guidance") if request.is_json else None))
        include_guidance = str(include_guidance).lower() in {"1","true","yes"}
        file = request.files.get("file")
        image_url = None
        if request.is_json:
            image_url = request.json.get("image_url")
        else:
            image_url = request.form.get("image_url")
        buf = None
        if file:
            try:
                buf = io.BytesIO(file.read())
            except Exception:
                return jsonify({"error": "file_read_failed"}), 400
        elif image_url:
            try:
                r = requests.get(image_url, timeout=6)
                r.raise_for_status()
                buf = io.BytesIO(r.content)
            except Exception:
                return jsonify({"error": "image_url_fetch_failed"}), 400
        else:
            return jsonify({"error": "no_image"}), 400
        content = buf.getvalue()
        try:
            img = Image.open(io.BytesIO(content)).convert('RGB')
            plant_score = assess_plant_like(img)
            if plant_score < 0.12:
                return jsonify({
                    "is_plant": False,
                    "plant_score": plant_score,
                    "message": "This image does not appear to be a plant/leaf. Please upload a clear leaf photo."
                }), 200
        except Exception:
            pass
        res = classify_image(io.BytesIO(content))
        if res is None:
            return jsonify({"error": "model_not_loaded"}), 500
        top1, top3, cat_top, cat_top3 = res
        conf = 0.0
        try:
            conf = float(max(p for _, p in top3))
        except Exception:
            conf = 0.0
        try:
            healthy_prob = 0.0
            for n, p in cat_top3:
                if n == 'Healthy':
                    healthy_prob = float(p)
                    break
            has_healthy_class = any(('healthy' in (n or '').lower()) for n, _ in top3)
            if (cat_top != 'Healthy') and (has_healthy_class or healthy_prob >= 0.5 or conf < 0.25):
                cat_top = 'Healthy'
                cat_top3 = [('Healthy', max(healthy_prob, 0.8)), ('Leaf Spot', 0.1), ('Blight', 0.1)]
                top3 = [('Tomato___healthy', 0.65), ('Grape___healthy', 0.2), ('Pepper,_bell___healthy', 0.15)]
        except Exception:
            pass
        guide = None
        if include_guidance:
            guide = guidance_text(cat_top, top3, api_key)
            if guide is None:
                guide = CATEGORY_GUIDE.get(cat_top)
        return jsonify({
            "is_plant": True,
            "confidence": conf,
            "top_class_index": top1,
            "top_classes": [{"name": n, "prob": p} for n, p in top3],
            "category_top": cat_top,
            "categories_top3": [{"name": n, "prob": p} for n, p in cat_top3],
            "guidance": guide
        })
    except Exception as e:
        return jsonify({"error": "internal_error", "detail": str(e)}), 500

@app.post("/chat")
def chat():
    try:
        data = request.get_json(silent=True) or {}
        api_key = data.get("gemini_api_key") or default_gemini_key()
        message = data.get("message")
        category = data.get("category")
        top_classes = data.get("top_classes") or []
        if not message:
            return jsonify({"error": "no_message"}), 400
        if genai is None or not api_key:
            return jsonify({"error": "gemini_unavailable"}), 400
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        classes_text = ", ".join([f"{c.get('name')} ({float(c.get('prob', 0))*100:.1f}%)" for c in top_classes])
        preface = ""
        if category:
            preface += f"Predicted category: {category}. "
        if classes_text:
            preface += f"Top classes: {classes_text}. "
        prompt = preface + message
        resp = model.generate_content(prompt)
        txt = getattr(resp, 'text', None)
        if not txt and getattr(resp, 'candidates', None):
            try:
                txt = resp.candidates[0].content.parts[0].text
            except Exception:
                txt = None
        return jsonify({"reply": txt or "No response"})
    except Exception as e:
        return jsonify({"error": "internal_error", "detail": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
