import os
import io
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

app = Flask(__name__)
CORS(app)

try:
    import google.generativeai as genai
except Exception:
    genai = None

HAS_ML = False
np = None
Image = None
keras = None

try:
    import numpy as np
except Exception:
    pass

try:
    from PIL import Image
except Exception:
    pass

try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow
    keras = tensorflow.keras
    HAS_ML = True
except Exception:
    keras = None
    HAS_ML = False

MODEL = None
MODEL_INFO = {}
CLASS_NAMES = []
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CATEGORY_OF = {
    'Apple___Apple_scab': 'Leaf Spot',
    'Apple___Black_rot': 'Blight',
    'Apple___Cedar_apple_rust': 'Rust',
    'Apple___healthy': 'Healthy',
    'Blueberry___healthy': 'Healthy',
    'Cherry___healthy': 'Healthy',
    'Cherry___Powdery_mildew': 'Blight',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': 'Leaf Spot',
    'Corn___Common_rust': 'Rust',
    'Corn___healthy': 'Healthy',
    'Corn___Northern_Leaf_Blight': 'Blight',
    'Grape___Black_rot': 'Blight',
    'Grape___Esca_(Black_Measles)': 'Blight',
    'Grape___healthy': 'Healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Leaf Spot',
    'Orange___Haunglongbing_(Citrus_greening)': 'Nutrient deficiency',
    'Peach___Bacterial_spot': 'Leaf Spot',
    'Peach___healthy': 'Healthy',
    'Pepper,_bell___Bacterial_spot': 'Leaf Spot',
    'Pepper,_bell___healthy': 'Healthy',
    'Potato___Early_blight': 'Blight',
    'Potato___healthy': 'Healthy',
    'Potato___Late_blight': 'Blight',
    'Raspberry___healthy': 'Healthy',
    'Soybean___healthy': 'Healthy',
    'Squash___Powdery_mildew': 'Blight',
    'Strawberry___healthy': 'Healthy',
    'Strawberry___Leaf_scorch': 'Blight',
    'Tomato___Bacterial_spot': 'Leaf Spot',
    'Tomato___Early_blight': 'Blight',
    'Tomato___healthy': 'Healthy',
    'Tomato___Late_blight': 'Blight',
    'Tomato___Leaf_Mold': 'Blight',
    'Tomato___Septoria_leaf_spot': 'Leaf Spot',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Blight',
    'Tomato___Target_Spot': 'Leaf Spot',
    'Tomato___Tomato_mosaic_virus': 'Blight',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Blight',
}

CATEGORY_GUIDE = {
    'Leaf Spot': "Remove infected leaves; improve airflow; avoid overhead watering; apply copper or chlorothalonil per label; sanitize tools; rotate crops.",
    'Blight': "Prune infected tissue; dispose, do not compost; avoid wet foliage; apply approved fungicide; use resistant varieties; rotate crops.",
    'Rust': "Remove rusted leaves; increase spacing; water at soil level; apply sulfur or copper per label; remove alternate hosts.",
    'Nutrient deficiency': "Add balanced NPK; use compost; ensure soil pH ~6–7; avoid overwatering; apply micronutrients if leaf veins stay green.",
    'Healthy': "Maintain spacing; water at roots; mulch; monitor weekly; keep tools clean; balanced fertilization."
}

def try_load_class_names():
    global CLASS_NAMES
    
    paths = [
        os.path.join(BASE_DIR, "disease_class_names.json"),
        os.path.join(os.getcwd(), "disease_class_names.json"),
        os.path.join(os.getcwd(), "class_names.json"),
        os.getenv("MODEL_CLASS_NAMES_JSON", "").strip(),
    ]
    
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                CLASS_NAMES = [str(x) for x in data]
                return True
            elif isinstance(data, dict) and "class_names" in data:
                CLASS_NAMES = [str(x) for x in data["class_names"]]
                return True
        except Exception:
            pass
    
    return False

def load_model():
    global MODEL, MODEL_INFO
    if MODEL is not None:
        return MODEL
    if not HAS_ML or keras is None:
        return None
    # try server folder JSON + H5
    try:
        json_path = os.path.join(BASE_DIR, "model.json")
        h5_path = os.path.join(BASE_DIR, "model.h5")
        if os.path.exists(json_path) and os.path.exists(h5_path):
            with open(json_path, "r", encoding="utf-8") as f:
                arch = f.read()
            m = keras.models.model_from_json(arch)
            m.load_weights(h5_path)
            MODEL = m
            MODEL_INFO = {"source": "json+h5", "json": json_path, "weights": h5_path}
            return MODEL
    except Exception:
        pass
    # try server folder .keras and .h5
    for p in [
        os.path.join(BASE_DIR, "model.keras"),
        os.path.join(BASE_DIR, "disease_model.keras"),
        os.path.join(BASE_DIR, "model.h5"),
        os.path.join(BASE_DIR, "disease_model.h5"),
    ]:
        if os.path.exists(p):
            try:
                MODEL = keras.models.load_model(p, compile=False)
                MODEL_INFO = {"source": "file", "path": p}
                return MODEL
            except Exception:
                pass
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
                MODEL = keras.models.load_model(env_keras, compile=False)
                MODEL_INFO = {"source": "keras", "path": env_keras}
                return MODEL
            except Exception:
                pass
        if env_h5 and os.path.exists(env_h5):
            try:
                MODEL = keras.models.load_model(env_h5, compile=False)
                MODEL_INFO = {"source": "h5", "path": env_h5}
                return MODEL
            except Exception:
                pass
    except Exception:
        pass
    candidates = [
        os.path.join(BASE_DIR, "disease_model.h5"),
        os.path.join(os.getcwd(), "disease_model.h5"),
        os.path.join(os.getcwd(), "model.h5"),
        os.path.join(os.getcwd(), "trained_model.h5"),
    ]
    
    for p in candidates:
        if os.path.exists(p):
            try:
                MODEL = keras.models.load_model(p, compile=False)
                MODEL_INFO = {"source": "h5", "path": p}
                return MODEL
            except Exception:
                pass
    
    return None

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
    if "powdery" in n:
        return "Blight"
    if ("huanglongbing" in n) or ("greening" in n):
        return "Nutrient deficiency"
    return "Blight"

def assess_plant_like(image):
    try:
        hsv = image.resize((256, 256)).convert('HSV')
        if np is not None:
            arr = np.array(hsv, dtype=np.uint8)
            H = arr[:,:,0].astype(np.float32) * (360.0/255.0)
            S = arr[:,:,1].astype(np.float32) / 255.0
            V = arr[:,:,2].astype(np.float32) / 255.0
            green = ((H>=35)&(H<=85)&(S>0.20)&(V>0.20)).mean()
            yellow = ((H>=20)&(H<=35)&(S>0.25)&(V>0.25)).mean()
            brown = (((H>=10)&(H<=30))&(S>0.20)&(V<0.45)).mean()
            sky = (((H>=180)&(H<=260))&(S<0.25)&(V>0.60)).mean()
        else:
            data = list(hsv.getdata())
            total = len(data) or 1
            g = y = b = s = 0
            for h,sat,val in data:
                hdeg = (h*360.0)/255.0
                S = sat/255.0
                V = val/255.0
                if (35<=hdeg<=85) and (S>0.20) and (V>0.20):
                    g += 1
                if (20<=hdeg<=35) and (S>0.25) and (V>0.25):
                    y += 1
                if (10<=hdeg<=30) and (S>0.20) and (V<0.45):
                    b += 1
                if (180<=hdeg<=260) and (S<0.25) and (V>0.60):
                    s += 1
            green = g/total
            yellow = y/total
            brown = b/total
            sky = s/total
        plant_score = float(green*0.6 + yellow*0.25 - sky*0.2 - brown*0.15)
        return max(0.0, plant_score)
    except Exception:
        return 0.5

def classify_image(file_like):
    try:
        image = Image.open(file_like).convert("RGB")
    except Exception:
        return None
    # Try ML if available
    if HAS_ML and keras is not None and np is not None and len(CLASS_NAMES) > 0:
        model = load_model()
        if model is not None:
            try:
                img128 = image.resize((128, 128))
                raw = np.array(img128, dtype=np.float32)
                norm = raw / 255.0
                arr_raw = np.expand_dims(raw, axis=0)
                arr_norm = np.expand_dims(norm, axis=0)
                def infer(x):
                    try:
                        return model.predict(x, verbose=0)[0]
                    except TypeError:
                        return model.predict(x)[0]
                probs_norm = infer(arr_norm)
                probs_raw = infer(arr_raw)
                probs = probs_norm if float(np.max(probs_norm)) >= float(np.max(probs_raw)) else probs_raw
                top_idx = int(np.argsort(probs)[::-1][0])
                top3_idx = np.argsort(probs)[::-1][:3]
                top3 = [(CLASS_NAMES[i], float(probs[i])) for i in top3_idx]
                cats = ['Leaf Spot', 'Blight', 'Rust', 'Nutrient deficiency', 'Healthy']
                cat_probs = {c: 0.0 for c in cats}
                for i, p in enumerate(probs):
                    if i < len(CLASS_NAMES):
                        c = category_for(CLASS_NAMES[i])
                        if c in cat_probs:
                            cat_probs[c] += float(p)
                cat_items = [(c, cat_probs[c]) for c in cats]
                cat_items.sort(key=lambda x: x[1], reverse=True)
                return top_idx, top3, cat_items[0][0], cat_items[:3]
            except Exception:
                pass
    # Heuristic fallback (works without ML)
    try:
        hsv = image.resize((256, 256)).convert('HSV')
        if np is not None:
            arr = np.array(hsv, dtype=np.uint8)
            H = arr[:,:,0].astype(np.float32) * (360.0/255.0)
            S = arr[:,:,1].astype(np.float32) / 255.0
            V = arr[:,:,2].astype(np.float32) / 255.0
            green = ((H>=35)&(H<=85)&(S>0.2)).mean()
            yellow = ((H>=20)&(H<=35)&(S>0.25)).mean()
            rusty = (((H>=0)&(H<=20))&(S>0.35)&(V>0.35)).mean()
            dark = (V<0.25).mean()
        else:
            data = list(hsv.getdata())
            total = len(data) or 1
            g=y=ru=d=0
            for h,sat,val in data:
                hdeg = (h*360.0)/255.0
                S = sat/255.0
                V = val/255.0
                if (35<=hdeg<=85) and (S>0.2):
                    g+=1
                if (20<=hdeg<=35) and (S>0.25):
                    y+=1
                if (0<=hdeg<=20) and (S>0.35) and (V>0.35):
                    ru+=1
                if V<0.25:
                    d+=1
            green = g/total
            yellow = y/total
            rusty = ru/total
            dark = d/total
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
            top3 = [('Corn___Common_rust', items[0][1]), ('Cherry___Powdery_mildew', items[1][1]), ('Apple___Cedar_apple_rust', items[2][1])]
        elif cat_top == 'Leaf Spot':
            top3 = [('Tomato___Septoria_leaf_spot', items[0][1]), ('Apple___Apple_scab', items[1][1]), ('Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', items[2][1])]
        elif cat_top == 'Nutrient deficiency':
            top3 = [('Orange___Haunglongbing_(Citrus_greening)', items[0][1]), ('Tomato___Tomato_Yellow_Leaf_Curl_Virus', items[1][1]), ('Potato___healthy', items[2][1])]
        elif cat_top == 'Healthy':
            top3 = [('Tomato___healthy', items[0][1]), ('Grape___healthy', items[1][1]), ('Pepper,_bell___healthy', items[2][1])]
        else:
            top3 = [('Tomato___Early_blight', items[0][1]), ('Potato___Early_blight', items[1][1]), ('Corn___Northern_Leaf_Blight', items[2][1])]
        return 0, top3, cat_top, cat_top3
    except Exception:
        return None

def default_gemini_key():
    k = os.getenv("GEMINI_API_KEY")
    if k:
        return k
    return None

def guidance_text(category, top3_classes, api_key):
    if genai is None or not api_key:
        return CATEGORY_GUIDE.get(category, "Consult a local agronomist for detailed guidance.")
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
    except Exception as e:
        return CATEGORY_GUIDE.get(category, "Consult a local agronomist for detailed guidance.")

try_load_class_names()

@app.get("/")
def home():
    return jsonify({
        "message": "Plant Disease Detection API",
        "status": "online",
        "ml_enabled": HAS_ML,
        "model_loaded": MODEL is not None
    })

@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "ml_available": HAS_ML,
        "model_loaded": MODEL is not None,
        "classes_count": len(CLASS_NAMES)
    })

@app.get("/status")
def status():
    return jsonify({
        "ml_available": HAS_ML,
        "model_loaded": MODEL is not None,
        "num_classes": len(CLASS_NAMES),
        "class_names_loaded": len(CLASS_NAMES) > 0,
        "model_info": MODEL_INFO,
        "remote_predict_url": os.getenv("REMOTE_PREDICT_URL", ""),
        "available_models": [p for p in [
            os.path.join(BASE_DIR, "model.h5"),
            os.path.join(BASE_DIR, "model.keras"),
            os.path.join(BASE_DIR, "model.json"),
            os.path.join(BASE_DIR, "disease_model.h5"),
            os.path.join(os.getcwd(), "disease_model.h5"),
            os.path.join(os.getcwd(), "model.h5"),
            os.path.join(os.getcwd(), "trained_model.h5"),
            os.getenv("MODEL_H5", "").strip(),
            os.getenv("MODEL_KERAS", "").strip(),
            os.getenv("MODEL_JSON", "").strip(),
            os.getenv("MODEL_WEIGHTS_H5", "").strip(),
        ] if p and os.path.exists(p)]
    })

@app.post("/predict")
def predict():
    try:
        api_key = request.form.get("gemini_api_key") or (request.json.get("gemini_api_key") if request.is_json else None) or default_gemini_key()
        include_guidance = (request.form.get("include_guidance") or (request.json.get("include_guidance") if request.is_json else None))
        include_guidance = str(include_guidance).lower() in {"1", "true", "yes"}
        
        file = request.files.get("file")
        image_url = request.form.get("image_url") if not request.is_json else request.json.get("image_url")
        
        if not file and not image_url:
            return jsonify({"error": "no_image"}), 400
        
        content = None
        if file:
            try:
                content = file.read()
            except Exception:
                return jsonify({"error": "file_read_failed"}), 400
        elif image_url:
            try:
                r = requests.get(image_url, timeout=6)
                r.raise_for_status()
                content = r.content
            except Exception:
                return jsonify({"error": "image_url_fetch_failed"}), 400
        
        if not content:
            return jsonify({"error": "no_content"}), 400
        
        buf = io.BytesIO(content)
        
        if Image is not None:
            try:
                image = Image.open(buf).convert('RGB')
                plant_score = assess_plant_like(image)
                
                if plant_score < 0.15:
                    return jsonify({
                        "is_plant": False,
                        "plant_score": plant_score,
                        "message": "Not a plant image. Please upload a clear leaf photo."
                    }), 200
            except Exception:
                pass
        
        buf.seek(0)
        res = classify_image(buf)
        if res is None:
            remote = os.getenv("REMOTE_PREDICT_URL", "").strip()
            if remote:
                try:
                    files = {'file': content}
                    rr = requests.post(remote, files=files, timeout=10)
                    if rr.status_code == 200:
                        return jsonify(rr.json())
                except Exception:
                    pass
        
        if res is None:
            return jsonify({"error": "classification_failed"}), 500
        
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
                top3 = [(t[0], 0.65 if 'healthy' in t[0].lower() else (0.2 if len(top3) > 1 else 0.15)) for t in top3[:1]] + [(t[0], (0.2 if len(top3) > 1 else 0.15)) for t in top3[1:]]
        except Exception:
            pass
        
        guide = None
        if include_guidance:
            guide = guidance_text(cat_top, top3, api_key)
        
        return jsonify({
            "is_plant": True,
            "confidence": conf,
            "top_class_index": top1,
            "top_classes": [{"name": n, "prob": p} for n, p in top3],
            "category_top": cat_top,
            "categories_top3": [{"name": n, "prob": p} for n, p in cat_top3],
            "guidance": guide
        })
    except Exception:
        return jsonify({"error": "internal_error"}), 500

@app.post("/chat")
def chat():
    try:
        data = request.get_json(silent=True) or {}
        message = data.get("message")
        
        if not message:
            return jsonify({"error": "no_message"}), 400
        
        api_key = data.get("gemini_api_key") or default_gemini_key()
        if genai is None or not api_key:
            category = data.get("category")
            return jsonify({"reply": CATEGORY_GUIDE.get(category, "Please ask your local agricultural extension.")}), 200
        
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            preface = ""
            category = data.get("category")
            if category:
                preface += f"Predicted category: {category}. "
            
            top_classes = data.get("top_classes") or []
            if top_classes:
                classes_text = ", ".join([f"{c.get('name')} ({float(c.get('prob', 0))*100:.1f}%)" for c in top_classes])
                preface += f"Top classes: {classes_text}. "
            
            resp = model.generate_content(preface + message)
            txt = getattr(resp, 'text', None) or (resp.candidates[0].content.parts[0].text if getattr(resp, 'candidates', None) else None)
            return jsonify({"reply": txt or "No response"})
        except Exception:
            return jsonify({"reply": "AI service unavailable."})
    except Exception:
        return jsonify({"error": "internal_error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
