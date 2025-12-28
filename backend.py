import os
import io
import json
import time
import re
import sqlite3
import threading
import base64
from flask import Flask, jsonify, request, Response, send_from_directory
from flask_cors import CORS
import requests

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

try:
    import cloudinary
    import cloudinary.uploader
    import cloudinary.api
    HAS_CLOUDINARY = True
except Exception:
    HAS_CLOUDINARY = False

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MAX_CLOUD_IMAGES = int(os.getenv("MAX_CLOUD_IMAGES", "1"))

if HAS_CLOUDINARY:
    try:
        cloudinary.config(
            cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", "dbtgh0xij"),
            api_key=os.getenv("CLOUDINARY_API_KEY", "827787627737112"),
            api_secret=os.getenv("CLOUDINARY_API_SECRET", "PtxKvOaoTKlkqW_afTaxDTcm1Uo")
        )
    except Exception:
        pass

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
    import serial
    from serial.tools import list_ports
except Exception:
    serial = None
    list_ports = None

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
SENSORS_DB_PATH = os.path.join(BASE_DIR, "sensors.db")
DB = None
SERIAL_PORT = os.getenv("ESP32_SERIAL_PORT", "").strip()
SERIAL_BAUD = int(os.getenv("ESP32_SERIAL_BAUD", "115200"))

LATEST_SENSOR_READING = {}
LATEST_SENSOR_READING_LOCK = threading.Lock()

LATEST_IMAGE = None
LATEST_IMAGE_LOCK = threading.Lock()

def ensure_db():
    global DB
    if DB is None:
        DB = sqlite3.connect(SENSORS_DB_PATH, check_same_thread=False)
        try:
            DB.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass
        DB.execute("CREATE TABLE IF NOT EXISTS sensor_readings (id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, temperature REAL, humidity REAL, soil INTEGER, rain INTEGER, light REAL)")
        try:
            DB.execute("ALTER TABLE sensor_readings ADD COLUMN light REAL")
        except Exception:
            pass
        DB.execute("CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT)")
        DB.execute("CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, image_url TEXT, disease_name TEXT, confidence REAL, guidance TEXT)")
        DB.commit()
    return DB

def set_config(key, value):
    conn = ensure_db()
    conn.execute("INSERT INTO config(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(key), str(value)))
    conn.commit()

def get_config(key, default=""):
    conn = ensure_db()
    cur = conn.execute("SELECT value FROM config WHERE key=?", (str(key),))
    row = cur.fetchone()
    return (row[0] if row else default)

def cleanup_old_cloud_images():
    if not HAS_CLOUDINARY:
        return
    try:
        resources = cloudinary.api.resources(resource_type="image", max_results=100)
        images = resources.get('resources', [])
        if len(images) > MAX_CLOUD_IMAGES:
            sorted_images = sorted(images, key=lambda x: x['created_at'], reverse=True)
            images_to_delete = sorted_images[MAX_CLOUD_IMAGES:]
            for img in images_to_delete:
                try:
                    cloudinary.uploader.destroy(img['public_id'])
                except Exception:
                    pass
    except Exception:
        pass

def upload_to_cloudinary(file_path):
    if not HAS_CLOUDINARY:
        return None
    try:
        cleanup_old_cloud_images() # Clean before upload to maintain limit
        result = cloudinary.uploader.upload(file_path, resource_type="auto")
        return result.get('secure_url')
    except Exception:
        return None

def latest_cloudinary_image_url():
    if not HAS_CLOUDINARY:
        return None
    try:
        resources = cloudinary.api.resources(resource_type="image", max_results=1, direction="desc") # simplified
        images = resources.get("resources", []) or []
        if not images:
            return None
        img = images[0] or {}
        return img.get("secure_url") or img.get("url")
    except Exception:
        return None

def store_prediction(image_url, disease_name, confidence, guidance=""):
    try:
        print(f"DEBUG: Storing prediction - Disease: {disease_name}, Confidence: {confidence}")
        conn = ensure_db()
        ts = int(time.time())
        conn.execute(
            "INSERT INTO predictions(ts, image_url, disease_name, confidence, guidance) VALUES(?, ?, ?, ?, ?)",
            (ts, image_url, disease_name, confidence, guidance)
        )
        conn.commit()
        print(f"DEBUG: Prediction stored successfully at TS={ts}")
    except Exception as e:
        print(f"DEBUG: Error in store_prediction: {e}")

def get_predictions(limit=20):
    conn = ensure_db()
    cur = conn.execute(
        "SELECT id, ts, image_url, disease_name, confidence, guidance FROM predictions ORDER BY ts DESC LIMIT ?",
        (limit,)
    )
    rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "ts": r[1],
            "image_url": r[2],
            "disease": r[3],
            "confidence": r[4],
            "guidance": r[5]
        }
        for r in rows
    ]

def latest_reading():
    conn = ensure_db()
    cur = conn.execute("SELECT ts, temperature, humidity, soil, rain, light FROM sensor_readings ORDER BY ts DESC LIMIT 1")
    row = cur.fetchone()
    if not row:
        return None
    ts = int(row[0])
    t = float(row[1] or 0.0)
    h = float(row[2] or 0.0)
    s = int(row[3] or 0)
    r = int(row[4] if row[4] is not None else 4095)
    l = 0.0
    if len(row) > 5 and row[5] is not None:
        try:
            l = float(row[5])
        except Exception:
            l = 0.0
    return {"ts": ts, "temperature": t, "humidity": h, "soil": s, "rain": r, "light": l}

def parse_html(html):
    try:
        temp_m = re.search(r"Temperature:\s*([\-\d\.]+)", html, re.I)
        hum_m = re.search(r"Humidity:\s*([\-\d\.]+)", html, re.I)
        soil_m = re.search(r"Soil\s*Moisture:\s*(\d+)", html, re.I)
        rain_m = re.search(r"Rain\s*Sensor:\s*(No\s*Rain|Rain)", html, re.I)
        t = float(temp_m.group(1)) if temp_m else 0.0
        h = float(hum_m.group(1)) if hum_m else 0.0
        s = int(soil_m.group(1)) if soil_m else 0
        rv = rain_m.group(1).strip().lower() if rain_m else "no rain"
        rain_val = 0 if rv == "rain" else 4095
        return {"temperature": t, "humidity": h, "soil": s, "rain": int(rain_val)}
    except Exception:
        return None

def fetch_from_esp32(base_url):
    u = str(base_url or "").strip()
    if not u:
        return None
    try:
        r = requests.get(u.rstrip("/") + "/api/sensors", timeout=5)
        if r.status_code == 200:
            try:
                j = r.json()
                t = float(j.get("temperature", 0.0))
                h = float(j.get("humidity", 0.0))
                s = int(j.get("soil", 0))
                rain_val = int(j.get("rain", 4095))
                return {"temperature": t, "humidity": h, "soil": s, "rain": rain_val}
            except Exception:
                pass
    except Exception:
        pass
    try:
        r = requests.get(u, timeout=5)
        if r.status_code == 200:
            parsed = parse_html(r.text or "")
            return parsed
    except Exception:
        pass
    return None

def parse_serial_text(txt):
    try:
        temp_m = re.search(r"Temperature:\s*([\-\d\.]+)", txt, re.I)
        hum_m = re.search(r"Humidity:\s*([\-\d\.]+)", txt, re.I)
        soil_m = re.search(r"Soil\s*Moisture:\s*(\d+)", txt, re.I)
        rain_m = re.search(r"Rain:\s*(No\s*Rain|Rain)", txt, re.I)
        t = float(temp_m.group(1)) if temp_m else None
        h = float(hum_m.group(1)) if hum_m else None
        s = int(soil_m.group(1)) if soil_m else None
        rv = rain_m.group(1).strip().lower() if rain_m else None
        if t is None or h is None or s is None or rv is None:
            return None
        raining = True if rv == "rain" else False
        return {"temperature": t, "humidity": h, "soil": s, "rain": raining}
    except Exception:
        return None

def find_serial_port():
    if SERIAL_PORT:
        return SERIAL_PORT
    try:
        if list_ports is not None:
            ports = [p.device for p in list_ports.comports()]
            for p in ports:
                if str(p).upper().startswith("COM"):
                    return p
            return ports[0] if ports else None
    except Exception:
        return None
    return None

def read_serial_loop(port_name):
    if serial is None:
        return
    try:
        ser = serial.Serial(port_name, SERIAL_BAUD, timeout=1)
    except Exception:
        return
    while True:
        try:
            raw = ser.readline()
            if not raw:
                continue
            line = None
            try:
                line = raw.decode(errors="ignore").strip()
            except Exception:
                continue
            rd = parse_serial_text(line)
            if rd:
                conn = ensure_db()
                ts = int(time.time())
                rain_raw = 0 if rd["rain"] else 4095
                conn.execute(
                    "INSERT INTO sensor_readings(ts, temperature, humidity, soil, rain, light) VALUES(?, ?, ?, ?, ?, ?)",
                    (ts, float(rd["temperature"]), float(rd["humidity"]), int(rd["soil"]), int(rain_raw), 0.0),
                )
                conn.commit()
        except Exception:
            time.sleep(0.5)

def start_serial_reader():
    try:
        p = find_serial_port()
        if p:
            t = threading.Thread(target=read_serial_loop, args=(p,), daemon=True)
            t.start()
    except Exception:
        pass

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

CATEGORY_GUIDE_UR = {
    'Leaf Spot': "متاثرہ پتوں کو ہٹا دیں؛ ہوا کی آمد و رفت بہتر بنائیں؛ اوپر سے پانی دینے سے گریز کریں؛ کاپر یا کلوروتھالونیل کا استعمال کریں؛ اوزار صاف کریں؛ فصلوں کو باری باری کاشت کریں۔",
    'Blight': "متاثرہ حصے کو کاٹ دیں؛ تلف کریں، کھاد نہ بنائیں؛ پتوں کو گیلا ہونے سے بچائیں؛ منظور شدہ پھپھوندی کش دوا استعمال کریں؛ مزاحم اقسام استعمال کریں؛ فصلوں کو باری باری کاشت کریں۔",
    'Rust': "زنگ آلود پتے ہٹا دیں؛ پودوں کے درمیان فاصلہ بڑھائیں؛ جڑوں میں پانی دیں؛ سلفر یا کاپر کا استعمال کریں؛ متبادل میزبان پودوں کو ہٹا دیں۔",
    'Nutrient deficiency': "متوازن NPK کھاد ڈالیں؛ نامیاتی کھاد استعمال کریں؛ مٹی کا pH 6-7 رکھیں؛ زیادہ پانی دینے سے گریز کریں؛ اگر پتوں کی رگیں سبز رہیں تو مائیکرو نیوٹرینٹس ڈالیں۔",
    'Healthy': "فاصلہ برقرار رکھیں؛ جڑوں میں پانی دیں؛ ملچنگ کریں؛ ہفتہ وار نگرانی کریں؛ اوزار صاف رکھیں؛ متوازن کھاد دیں۔"
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

def language_name(code):
    c = str(code or "").lower()
    if c == "ur":
        return "Urdu"
    if c == "en":
        return "English"
    return "English"

def guidance_text(category, top3_classes, api_key, language="en"):
    default_msg = "Consult a local agronomist for detailed guidance."
    guide_map = CATEGORY_GUIDE
    if language == "ur":
        default_msg = "تفصیلی رہنمائی کے لیے مقامی زرعی ماہر سے مشورہ کریں۔"
        guide_map = CATEGORY_GUIDE_UR

    if genai is None or not api_key:
        return guide_map.get(category, default_msg)
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        classes_text = ", ".join([f"{n} ({p*100:.1f}%)" for n, p in top3_classes])
        lang_name = "Urdu" if language == "ur" else "English"
        prompt = (
            f"You are an agronomy assistant. The image was predicted as '{category}'. "
            f"Top classes: {classes_text}. Provide a concise summary in {lang_name}: 1) disease overview, 2) key symptoms, 3) immediate actions, 4) low-cost treatment plan, 5) prevention. "
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
        return guide_map.get(category, default_msg)

try_load_class_names()
ensure_db()

@app.post("/esp32/register")
def esp32_register():
    try:
        data = request.get_json(silent=True) or {}
        base_url = str(data.get("base_url") or "").strip()
        if not base_url:
            return jsonify({"error": "no_base_url"}), 400
        set_config("esp32_url", base_url)
        return jsonify({"ok": True, "esp32_url": base_url})
    except Exception:
        return jsonify({"error": "internal_error"}), 500

@app.post("/sensors/pull")
def sensors_pull():
    try:
        data = request.get_json(silent=True) or {}
        base_url = str(data.get("base_url") or get_config("esp32_url", "") or os.getenv("ESP32_SENSOR_URL", "")).strip()
        if not base_url:
            return jsonify({"error": "no_base_url"}), 400
        reading = fetch_from_esp32(base_url)
        if not reading:
            return jsonify({"error": "fetch_failed"}), 502
        conn = ensure_db()
        ts = int(time.time())
        conn.execute(
            "INSERT INTO sensor_readings(ts, temperature, humidity, soil, rain, light) VALUES(?, ?, ?, ?, ?, ?)",
            (ts, float(reading["temperature"]), float(reading["humidity"]), int(reading["soil"]), int(reading.get("rain", 4095)), float(reading.get("light", 0.0)))
        )
        conn.commit()
        return jsonify({"ts": ts, **reading})
    except Exception:
        return jsonify({"error": "internal_error"}), 500

@app.get("/sensors/latest")
def sensors_latest():
    try:
        with LATEST_SENSOR_READING_LOCK:
            if LATEST_SENSOR_READING:
                return jsonify(LATEST_SENSOR_READING)
        
        lr = latest_reading()
        if lr:
            return jsonify(lr)
        
        esp32_url = get_config("esp32_url", "").strip() or os.getenv("ESP32_SENSOR_URL", "").strip()
        
        if esp32_url:
            try:
                reading = fetch_from_esp32(esp32_url)
                if reading:
                    ts = int(time.time())
                    return jsonify({"ts": ts, **reading})
            except Exception as e:
                print(f"ESP32 fetch error: {e}")
        
        return jsonify({"error": "no_data", "message": "Waiting for first sensor reading from ESP32"}), 404
    except Exception:
        return jsonify({"error": "internal_error"}), 500

@app.get("/sensors/config")
def sensors_config():
    try:
        return jsonify({
            "esp32_url": get_config("esp32_url", ""),
            "esp32_camera_url": get_config("esp32_camera_url", ""),
        })
    except Exception:
        return jsonify({"error": "internal_error"}), 500

@app.post("/sensors/store")
@app.post("/sensors/update")
def store_sensor_data():
    try:
        data = request.get_json(silent=True)
        if data is None:
            # Try to parse manually if Content-Type was missing
            try:
                data = json.loads(request.data.decode('utf-8'))
            except Exception:
                data = None
        
        if not data:
            print(f"DEBUG: /sensors/store received no data. Raw: {request.data}")
            return jsonify({"error": "no_data", "received": str(request.data)}), 400
        
        print(f"DEBUG: /sensors/store received: {data}")
        
        t = float(data.get("t") or data.get("temperature") or 0.0)
        h = float(data.get("h") or data.get("humidity") or 0.0)
        s = int(data.get("s") or data.get("soil") or 0)
        r = int(data.get("r") or data.get("rain") or 4095)
        l = float(data.get("l") or data.get("light") or 0.0)
        ts = int(data.get("ts") or time.time())

        conn = ensure_db()
        conn.execute(
            "INSERT INTO sensor_readings(ts, temperature, humidity, soil, rain, light) VALUES(?, ?, ?, ?, ?, ?)",
            (ts, t, h, s, r, l)
        )
        conn.commit()
        
        with LATEST_SENSOR_READING_LOCK:
            LATEST_SENSOR_READING.clear()
            LATEST_SENSOR_READING.update({
                "ts": ts,
                "temperature": t,
                "humidity": h,
                "soil": s,
                "rain": r,
                "light": l
            })

        print(f"Stored sensor data: TS={ts}, T={t}, H={h}, S={s}, R={r}, L={l}")

        return jsonify({"ok": True, "ts": ts}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/sensors/push")
def sensors_push():
    try:
        data = request.get_json(silent=True) or {}
        t = float(data.get("temperature", 0.0))
        h = float(data.get("humidity", 0.0))
        s = int(data.get("soil", 0))
        r = int(data.get("rain", 4095))
        l = float(data.get("light", 0.0))
        ts = int(data.get("ts", int(time.time())))
        
        with LATEST_SENSOR_READING_LOCK:
            LATEST_SENSOR_READING.clear()
            LATEST_SENSOR_READING.update({
                "ts": ts,
                "temperature": t,
                "humidity": h,
                "soil": s,
                "rain": r,
                "light": l
            })
        
        conn = ensure_db()
        conn.execute(
            "INSERT INTO sensor_readings(ts, temperature, humidity, soil, rain, light) VALUES(?, ?, ?, ?, ?, ?)",
            (ts, t, h, s, r, l),
        )
        conn.commit()
        return jsonify({"ok": True})
    except Exception:
        return jsonify({"error": "internal_error"}), 500

@app.post("/sensors/register")
def sensors_register():
    try:
        data = request.get_json(silent=True) or {}
        sensor_url = str(data.get("url") or "").strip()
        if not sensor_url:
            return jsonify({"error": "no_url"}), 400
        set_config("esp32_url", sensor_url)
        return jsonify({"ok": True, "sensor_url": sensor_url})
    except Exception:
        return jsonify({"error": "internal_error"}), 500

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
    lr = latest_reading()
    now = int(time.time())
    sensor_ok = False
    if lr and abs(now - int(lr.get("ts", 0))) <= 600:
        sensor_ok = True
    return jsonify({
        "ok": True,
        "ml_available": HAS_ML,
        "model_loaded": MODEL is not None,
        "classes_count": len(CLASS_NAMES),
        "sensor_available": sensor_ok
    })

@app.get("/status")
def status():
    return jsonify({
        "ml_available": HAS_ML,
        "model_loaded": MODEL is not None,
        "num_classes": len(CLASS_NAMES),
        "class_names_loaded": len(CLASS_NAMES) > 0,
        "model_info": MODEL_INFO,
        "esp32_url": get_config("esp32_url", ""),
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

@app.get("/ping")
def ping():
    try:
        return jsonify({"ok": True})
    except Exception:
        return jsonify({"ok": False}), 500

@app.post("/predict")
def predict():
    try:
        json_data = request.get_json(silent=True) or {}
        form_data = request.form or {}
        
        api_key = form_data.get("gemini_api_key") or json_data.get("gemini_api_key") or default_gemini_key()
        include_guidance = form_data.get("include_guidance") or json_data.get("include_guidance")
        include_guidance = str(include_guidance or "").lower() in {"1", "true", "yes"}
        language = str(form_data.get("language") or json_data.get("language") or "en").lower()
        
        content = None
        
        file = request.files.get("file")
        image_url = form_data.get("image_url") or json_data.get("image_url")

        if not file and not image_url:
            return jsonify({"error": "no_image"}), 400
        
        if file:
            try:
                content = file.read()
            except Exception:
                return jsonify({"error": "file_read_failed"}), 400
        elif image_url:
            try:
                print(f"Fetching image from: {image_url}")
                r = requests.get(image_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                r.raise_for_status()
                content = r.content
            except Exception as e:
                print(f"Fetch failed: {e}")
                return jsonify({"error": "image_url_fetch_failed", "details": str(e)}), 400
        
        if not content:
            return jsonify({"error": "no_content"}), 400
        
        buf = io.BytesIO(content)
        
        if Image is not None:
            try:
                image = Image.open(buf).convert('RGB')
                plant_score = assess_plant_like(image)
                
                if plant_score < 0.15:
                    msg = "Not a plant image. Please upload a clear leaf photo."
                    if language == "ur":
                        msg = "یہ پودے کی تصویر نہیں ہے۔ براہ کرم پتے کی صاف تصویر اپ لوڈ کریں۔"
                    return jsonify({
                        "is_plant": False,
                        "plant_score": plant_score,
                        "message": msg
                    }), 200
            except Exception:
                pass
        
        buf.seek(0)
        res = classify_image(buf)
        
        # If local classification fails, try remote
        if res is None:
            remote = os.getenv("REMOTE_PREDICT_URL", "").strip()
            if remote:
                try:
                    print(f"DEBUG: Trying remote prediction at {remote}")
                    files = {"file": ("image.jpg", content, "image/jpeg")}
                    rr = requests.post(remote, files=files, timeout=10)
                    if rr.status_code == 200:
                        remote_res = rr.json()
                        # Capture values from remote response for storage
                        cat_top = remote_res.get("category_top") or remote_res.get("disease") or "Unknown"
                        conf = float(remote_res.get("confidence") or 0.0)
                        guide = remote_res.get("guidance") or ""
                        
                        # Store and return
                        store_prediction(image_url or "", cat_top, conf, guide)
                        return jsonify(remote_res)
                except Exception as e:
                    print(f"DEBUG: Remote prediction failed: {e}")
        
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
            guide = guidance_text(cat_top, top3, api_key, language)
        
        # If we have a file but no image_url, try to upload to get a URL for history
        if not image_url and file and HAS_CLOUDINARY:
            try:
                filename = f"pred_{int(time.time())}.jpg"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                with open(filepath, "wb") as f:
                    f.write(content)
                image_url = upload_to_cloudinary(filepath)
                try: os.remove(filepath)
                except: pass
            except Exception as e:
                print(f"Failed to upload prediction image: {e}")

        # Always store the prediction for history
        try:
            store_prediction(image_url or "", cat_top, conf, guide or "")
        except Exception as e:
            print(f"Failed to store prediction: {e}")
        
        return jsonify({
            "is_plant": True,
            "confidence": conf,
            "top_class_index": top1,
            "top_classes": [{"name": n, "prob": p} for n, p in top3],
            "category_top": cat_top,
            "categories_top3": [{"name": n, "prob": p} for n, p in cat_top3],
            "guidance": guide,
            "image_url": image_url # Return the URL if we generated one
        })
    except Exception as e:
        print(f"Predict error: {e}")
        return jsonify({"error": "internal_error", "details": str(e)}), 500

@app.get("/predictions")
def get_stored_predictions():
    try:
        limit = request.args.get("limit", 20, type=int)
        predictions = get_predictions(limit=limit)
        return jsonify({"predictions": predictions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/chat")
def chat():
    try:
        data = request.get_json(silent=True) or {}
        message = data.get("message")
        
        if not message:
            return jsonify({"error": "no_message"}), 400
        
        api_key = data.get("gemini_api_key") or default_gemini_key()
        # 'language' param passed from Flutter ('en' or 'ur')
        req_language = str(data.get("language") or "en").lower()
        
        if genai is None or not api_key:
            category = data.get("category")
            guide_map = CATEGORY_GUIDE
            default_msg = "Please ask your local agricultural extension."
            if req_language == "ur":
                guide_map = CATEGORY_GUIDE_UR
                default_msg = "براہ کرم اپنے مقامی زرعی توسیعی مرکز سے پوچھیں۔"
            
            return jsonify({"reply": guide_map.get(category, default_msg)}), 200
        
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            
            preface = ""
            category = data.get("category")
            if category:
                preface += f"Predicted category: {category}. "
            
            top_classes = data.get("top_classes") or []
            if top_classes:
                classes_text = ", ".join([f"{c.get('name')} ({float(c.get('prob', 0))*100:.1f}%)" for c in top_classes])
                preface += f"Top classes: {classes_text}. "
            
            # Explicit instruction for language
            target_lang_name = "Urdu" if req_language == "ur" else "English"
            preface = f"You are a helpful agriculture assistant for farmers. Respond in {target_lang_name}. " + preface
            
            resp = model.generate_content(preface + message)
            txt = getattr(resp, 'text', None) or (resp.candidates[0].content.parts[0].text if getattr(resp, 'candidates', None) else None)
            return jsonify({"reply": txt or "No response"})
        except Exception:
            err_msg = "AI service unavailable."
            if req_language == "ur":
                err_msg = "AI سروس دستیاب نہیں ہے۔"
            return jsonify({"reply": err_msg})
    except Exception:
        return jsonify({"error": "internal_error"}), 500

@app.post("/translate")
def translate():
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text")
        target = str(data.get("target") or "ur").lower()
        if not text:
            return jsonify({"error": "no_text"}), 400
        api_key = data.get("gemini_api_key") or default_gemini_key()
        if genai is None or not api_key:
            return jsonify({"translated": text}), 200
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            lang = "Urdu" if target == "ur" else "English"
            resp = model.generate_content(f"Translate to {lang}. Only translation:\n\n{text}")
            txt = getattr(resp, 'text', None) or (resp.candidates[0].content.parts[0].text if getattr(resp, 'candidates', None) else None)
            return jsonify({"translated": txt or text})
        except Exception:
            return jsonify({"translated": text})
    except Exception:
        return jsonify({"error": "internal_error"}), 500

@app.post("/camera/register")
def camera_register():
    try:
        data = request.get_json(silent=True) or {}
        url = str(data.get("url") or "").strip()
        if not url:
            return jsonify({"error": "no_url"}), 400
        set_config("esp32_camera_url", url)
        return jsonify({"ok": True, "camera_url": url})
    except Exception:
        return jsonify({"error": "internal_error"}), 500

@app.get("/camera/latest")
def camera_latest():
    try:
        global LATEST_IMAGE
        cam_url = get_config("esp32_camera_url", "")
        latest_img = None
        with LATEST_IMAGE_LOCK:
            latest_img = LATEST_IMAGE

        if not latest_img:
            cached = str(get_config("latest_image_url", "") or "").strip()
            latest_img = cached if cached else None

        if not latest_img:
            latest_img = latest_cloudinary_image_url()
            if latest_img:
                set_config("latest_image_url", latest_img)
                with LATEST_IMAGE_LOCK:
                    LATEST_IMAGE = latest_img

        return jsonify({
            "stream_url": cam_url,
            "latest_image_url": latest_img,
            "image_url": latest_img
        })
    except Exception:
        return jsonify({"error": "internal_error"}), 500

@app.post("/camera/upload")
def camera_upload():
    try:
        if not HAS_CLOUDINARY:
            return jsonify({"error": "cloudinary_not_configured"}), 501
        
        content = request.data
        if not content:
            return jsonify({"error": "no_content"}), 400

        cleanup_old_cloud_images()

        filename = f"cam_{int(time.time())}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        with open(filepath, "wb") as f:
            f.write(content)
            
        print(f"Uploading file: {filepath}")
        url = upload_to_cloudinary(filepath)
        print(f"Upload result: {url}")
        
        # Cleanup local
        try:
            os.remove(filepath)
        except:
            pass
            
        if url:
            with LATEST_IMAGE_LOCK:
                global LATEST_IMAGE
                LATEST_IMAGE = url

            set_config("latest_image_url", url)
            
            return jsonify({"ok": True, "url": url}), 200
        else:
            return jsonify({"error": "upload_failed"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/upload")
def upload_alias():
    return camera_upload()

@app.get("/latest")
def latest_alias():
    # Return format matching camera_latest but with "image_url" key for compatibility if needed
    try:
        res = camera_latest()
        if res.status_code == 200:
            data = res.get_json()
            # If camera_latest returns latest_image_url, map it to image_url for testing_cam compat
            if "latest_image_url" in data:
                data["image_url"] = data["latest_image_url"]
            return jsonify(data)
        return res
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/sensors/history")
def sensors_history_endpoint():
    try:
        limit = int(request.args.get("limit", 50))
        conn = ensure_db()
        cur = conn.execute(
            "SELECT ts, temperature, humidity, soil, rain, light FROM sensor_readings ORDER BY ts DESC LIMIT ?",
            (limit,)
        )
        rows = cur.fetchall()
        results = []
        for r in rows:
            ts = int(r[0])
            t = float(r[1] or 0.0)
            h = float(r[2] or 0.0)
            s = int(r[3] or 0)
            rain_val = int(r[4] if r[4] is not None else 4095)
            l = 0.0
            if len(r) > 5 and r[5] is not None:
                try:
                    l = float(r[5])
                except Exception:
                    l = 0.0
            results.append({
                "ts": ts,
                "temperature": t,
                "humidity": h,
                "soil": s,
                "rain": rain_val,
                "light": l
            })
        return jsonify({"readings": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/predictions/history")
def predictions_history():
    try:
        limit = int(request.args.get("limit", 20))
        predictions = get_predictions(limit)
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/ping")
def ping():
    return jsonify({"status": "ok", "time": int(time.time())})

if __name__ == "__main__":
    start_serial_reader()
    app.run(host="0.0.0.0", port=5000, debug=True)
