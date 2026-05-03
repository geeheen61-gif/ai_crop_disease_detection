import os
import io
import json
import time
import re
import sqlite3
import threading
import base64
import ssl
import urllib3
from flask import Flask, jsonify, request, Response, send_from_directory
from flask_cors import CORS
import requests
try:
    from twilio.twiml.messaging_response import MessagingResponse
    from twilio.rest import Client
    HAS_TWILIO = True
except ImportError:
    HAS_TWILIO = False

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
app.url_map.strict_slashes = False # Allow trailing slashes (e.g. /store and /store/ are same)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"], "allow_headers": ["Content-Type", "Authorization"]}})

# Disable InsecureRequestWarning for development when verify=False is used
try:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MAX_CLOUD_IMAGES = int(os.getenv("MAX_CLOUD_IMAGES", "1"))

# ================= FIREBASE CONFIG =================
# If you want the backend to fetch from Firebase:
FIREBASE_URL = os.getenv("FIREBASE_URL", "https://new-project-27194-default-rtdb.firebaseio.com/").strip()
if not FIREBASE_URL.endswith("/"):
    FIREBASE_URL += "/"
    
FIREBASE_SECRET = os.getenv("FIREBASE_SECRET", "").strip()

# ================= TWILIO CONFIG =================
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")
USER_PHONE_NUMBER = os.getenv("USER_PHONE_NUMBER", "")

if HAS_CLOUDINARY:
    try:
        cloudinary.config(
            cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", ""),
            api_key=os.getenv("CLOUDINARY_API_KEY", ""),
            api_secret=os.getenv("CLOUDINARY_API_SECRET", "")
        )
    except Exception:
        pass

HAS_ML = False




try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    import serial
    from serial.tools import list_ports
except Exception:
    serial = None
    list_ports = None

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

from flask import g

def get_db():
    """Provides a thread-safe database connection for the current request context."""
    if 'db' not in g:
        g.db = sqlite3.connect(SENSORS_DB_PATH, check_same_thread=False)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error):
    """Closes the database connection at the end of each request."""
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    """Initializes the database schema if it doesn't exist."""
    with app.app_context():
        conn = sqlite3.connect(SENSORS_DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                ts INTEGER, 
                temperature REAL, 
                humidity REAL, 
                soil INTEGER, 
                rain INTEGER, 
                light REAL
            )
        """)
        # Ensure 'light' column exists in older schemas
        try:
            conn.execute("ALTER TABLE sensor_readings ADD COLUMN light REAL")
        except Exception:
            pass
        conn.execute("CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, image_url TEXT, disease_name TEXT, confidence REAL, guidance TEXT)")
        conn.commit()
        conn.close()
        print(f"Database initialized at {SENSORS_DB_PATH}")

def set_config(key, value):
    conn = get_db()
    conn.execute("INSERT INTO config(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(key), str(value)))
    conn.commit()

def get_config(key, default=""):
    conn = get_db()
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
        print("DEBUG: Cloudinary NOT configured")
        return None, "Cloudinary not configured"
    try:
        cleanup_old_cloud_images() # Clean before upload to maintain limit
        print(f"DEBUG: Uploading {file_path} to Cloudinary...")
        result = cloudinary.uploader.upload(file_path, resource_type="auto")
        print(f"DEBUG: Upload success: {result}")
        return result.get('secure_url'), None
    except Exception as e:
        print(f"DEBUG: Cloudinary Exception: {e}")
        return None, str(e)

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
        ts = int(time.time())
        conn = get_db()
        conn.execute(
            "INSERT INTO predictions(ts, image_url, disease_name, confidence, guidance) VALUES(?, ?, ?, ?, ?)",
            (ts, image_url, disease_name, confidence, guidance)
        )
        conn.commit()
    except Exception as e:
        print(f"Error storing prediction: {e}")

def get_predictions(limit=20):
    conn = get_db()
    cur = conn.execute(
        "SELECT id, ts, image_url, disease_name, confidence, guidance FROM predictions ORDER BY ts DESC LIMIT ?",
        (limit,)
    )
    rows = cur.fetchall()
    return [
        {
            "id": r['id'],
            "ts": r['ts'],
            "image_url": r['image_url'],
            "disease": r['disease_name'], # Corrected mapping for Flutter
            "confidence": r['confidence'],
            "guidance": r['guidance']
        }
        for r in rows
    ]

def fetch_firebase_latest():
    """Fetches the most recent sensor reading from Firebase."""
    try:
        if not FIREBASE_URL or "firebaseio.com" not in FIREBASE_URL:
            return None
            
        # Firebase REST API to get the last entry
        # orderBy="$key"&limitToLast=1 gives the most recently added node (time-based keys)
        url = f"{FIREBASE_URL}sensors.json?orderBy=\"$key\"&limitToLast=1"
        if FIREBASE_SECRET:
            url += f"&auth={FIREBASE_SECRET}"
            
        # print(f"DEBUG: Fetching from Firebase: {url}")
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data and isinstance(data, dict):
                # result is {"-Key": {data...}}
                key = list(data.keys())[0]
                val = data[key]
                
                # Handle timestamp: Firebase sends ms, we usually want sec
                ts = val.get("ts", 0)
                if isinstance(ts, dict): # .sv: timestamp placeholder?
                    ts = int(time.time())
                elif ts > 1000000000000: # It's in ms
                    ts = int(ts / 1000)
                
                return {
                    "ts": ts,
                    "temperature": float(val.get("temperature", 0.0)),
                    "humidity": float(val.get("humidity", 0.0)),
                    "soil": int(val.get("soil", 0)),
                    "rain": int(val.get("rain", 4095)),
                    "light": float(val.get("light", 0.0))
                }
    except Exception as e:
        print(f"Firebase fetch error: {e}")
    return None

def latest_reading():
    """Retrieves the most recent sensor reading from Firebase (preferred) or database."""
    # 1. Try Firebase first
    fb_data = fetch_firebase_latest()
    if fb_data:
        # Optional: Cache it to local DB for history/redundancy
        try:
             # Check if we already have this ts to avoid dupes? 
             # For now, just return it. The frontend polls this.
             pass
        except:
             pass
        return fb_data

    # 2. Fallback to local DB
    try:
        conn = get_db()
        cur = conn.execute("SELECT ts, temperature, humidity, soil, rain, light FROM sensor_readings ORDER BY ts DESC LIMIT 1")
        row = cur.fetchone()
        
        if not row:
            return {
                "ts": int(time.time()),
                "temperature": 0.0,
                "humidity": 0.0,
                "soil": 0,
                "rain": 4095,
                "light": 0.0,
                "is_default": True
            }
            
        return {
            "ts": int(row['ts']),
            "temperature": float(row['temperature'] or 0.0),
            "humidity": float(row['humidity'] or 0.0),
            "soil": int(row['soil'] or 0),
            "rain": int(row['rain'] if row['rain'] is not None else 4095),
            "light": float(row['light'] or 0.0)
        }
    except Exception as e:
        print(f"Error fetching latest reading: {e}")
        return None

def send_sms_notification(body):
    """Sends an SMS using Twilio."""
    if not HAS_TWILIO:
        print("DEBUG: Twilio not installed, cannot send SMS")
        return False
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=body,
            from_=TWILIO_FROM_NUMBER,
            to=USER_PHONE_NUMBER
        )
        print(f"DEBUG: SMS sent successfully, SID: {message.sid}")
        return True
    except Exception as e:
        print(f"DEBUG: Failed to send SMS: {e}")
        return False

def check_sensors_and_notify_loop():
    """Background loop to check sensors and send notifications every 3 hours if needed."""
    print("DEBUG: Sensor notification loop started")
    while True:
        try:
            # We need an app context for get_db() and get_config()
            with app.app_context():
                reading = latest_reading()
                if reading and not reading.get("is_default"):
                    soil = reading.get("soil", 0)
                    rain = reading.get("rain", 4095)
                    
                    # Conditions: Soil moisture > 2800 OR Rain sensor value is low (e.g. < 500)
                    if soil > 2800 or rain < 500:
                        last_sent = float(get_config("last_sms_sent_ts", 0))
                        now = time.time()
                        
                        # 3 hours gap (10800 seconds)
                        if now - last_sent > 10800:
                            msg = f"⚠️ Alert: Unusual sensor values detected!\n"
                            if soil > 2800:
                                msg += f"- Soil Moisture: {soil} (High)\n"
                            if rain < 500:
                                msg += f"- Rain detected (Value: {rain})\n"
                            msg += f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}"
                            
                            if send_sms_notification(msg):
                                set_config("last_sms_sent_ts", now)
            
        except Exception as e:
            print(f"DEBUG: Error in sensor notification loop: {e}")
        
        # Check every 5 minutes
        time.sleep(300)



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
            line = raw.decode(errors="ignore").strip()
            rd = parse_serial_text(line)
            if rd:
                # Inside thread, we use a dedicated connection
                with sqlite3.connect(SENSORS_DB_PATH) as conn:
                    ts = int(time.time())
                    rain_raw = 0 if rd["rain"] else 4095
                    conn.execute(
                        "INSERT INTO sensor_readings(ts, temperature, humidity, soil, rain, light) VALUES(?, ?, ?, ?, ?, ?)",
                        (ts, float(rd["temperature"]), float(rd["humidity"]), int(rd["soil"]), int(rain_raw), 0.0),
                    )
                    conn.commit()
        except Exception:
            time.sleep(1)

def start_serial_reader():
    # Only run serial reader if physically on a machine (Local)
    # Render environment specifically sets the RENDER variable
    if os.environ.get("RENDER"):
         return # Skip on Render
    try:
        p = find_serial_port()
        if p:
            t = threading.Thread(target=read_serial_loop, args=(p,), daemon=True)
            t.start()
            print(f"Serial reader started on {p}")
    except Exception:
        pass

def start_sensor_notifier():
    """Starts the background thread for sensor notifications."""
    t = threading.Thread(target=check_sensors_and_notify_loop, daemon=True)
    t.start()
    print("Sensor notifier thread started")

start_serial_reader()
start_sensor_notifier()



def classify_image_with_gemini(content, api_key, language="en"):
    """Uses Gemini Vision API to detect plant disease from image content."""
    if genai is None or not api_key:
        return None
    
    try:
        genai.configure(api_key=api_key)
        # Safety settings to avoid blocking common leaf images
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        model = genai.GenerativeModel('gemini-2.5-flash', safety_settings=safety_settings) 
        
        # Prepare the image for Gemini
        image_parts = [
            {
                "mime_type": "image/jpeg",
                "data": content
            }
        ]
        
        lang_name = "Urdu" if language == "ur" else "English"
        prompt = (
            f"Act as an expert agronomist. Analyze this plant leaf image and provide a JSON response with the following fields: "
            f"1) 'is_plant' (boolean): whether the image is a plant leaf, "
            f"2) 'disease_name' (string): the specific disease or 'Healthy', "
            f"3) 'category' (string): a general category (e.g., Fungal, Bacterial, Viral, Nutrient, Healthy), "
            f"4) 'confidence' (float): 0 to 1, "
            f"5) 'guidance' (string): a detailed, expert treatment and prevention guide in {lang_name}. "
            f"Provide ONLY the raw JSON object."
        )
        
        response = model.generate_content([prompt, image_parts[0]])
        
        # Extract JSON from response using regex for better robustness
        text = response.text.strip()
        print(f"DEBUG: Gemini Raw Response: {text}")
        
        # Try to find JSON block
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)
        
        data = json.loads(text)
        return data
    except Exception as e:
        print(f"Gemini detection error: {str(e)}")
        # Return the error string so it can be reported
        return {"error": str(e)}

# Custom classification code removed in favor of Gemini API as per request.

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
    if language == "ur":
        default_msg = "تفصیلی رہنمائی کے لیے مقامی زرعی ماہر سے مشورہ کریں۔"

    if genai is None or not api_key:
        return default_msg
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        lang_name = "Urdu" if language == "ur" else "English"
        prompt = (
            f"You are an agronomy assistant. The image was identified as '{category}'. "
            f"Provide a concise summary in {lang_name}: 1) disease overview, 2) key symptoms, 3) immediate actions, 4) low-cost treatment plan, 5) prevention. "
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
        return default_msg


init_db()





@app.route("/sensors/latest", methods=["GET"])
def sensors_latest():
    """Returns the latest sensor data, preferring database for consistency across workers."""
    try:
        # print("DEBUG: fetching sensors/latest")
        # 1. Try Database first (consistent across multi-worker setups)
        lr = latest_reading()
        if lr and not lr.get("is_default"):
            return jsonify(lr), 200
            
        # 2. Check in-memory cache as fallback
        with LATEST_SENSOR_READING_LOCK:
            if LATEST_SENSOR_READING:
                return jsonify(LATEST_SENSOR_READING), 200
        
        # 3. Last fallback: return dummy data (prevents 404)
        ts = int(time.time())
        return jsonify({
            "ts": ts,
            "temperature": 0.0,
            "humidity": 0.0,
            "soil": 0,
            "rain": 4095,
            "light": 0.0,
            "message": "Waiting for sensor data",
            "is_default": True
        }), 200
    except Exception as e:
        print(f"Error in sensors_latest: {e}")
        return jsonify({"error": "internal_error", "details": str(e)}), 500

@app.route("/sensors/config", methods=["GET"])
def sensors_config():
    try:
        return jsonify({
            "esp32_url": get_config("esp32_url", ""),
            "esp32_camera_url": get_config("esp32_camera_url", ""),
        })
    except Exception:
        return jsonify({"error": "internal_error"}), 500

@app.route("/sensors/store", methods=["POST"])
@app.route("/sensors/update", methods=["POST"])
def store_sensor_data():
    try:
        data = None
        
        # Debug: Log all request info
        print(f"[DEBUG] Request method: {request.method}")
        print(f"[DEBUG] Content-Type: {request.content_type}")
        print(f"[DEBUG] is_json: {request.is_json}")
        print(f"[DEBUG] Raw data length: {len(request.data)}")
        
        # Robust parsing for JSON
        if request.is_json:
            data = request.get_json(silent=True)
            print(f"[DEBUG] Parsed from is_json: {data}")
        
        if not data:
            try:
                raw_text = request.get_data(as_text=True)
                print(f"[DEBUG] Raw text: {raw_text[:200] if raw_text else 'empty'}")
                if raw_text:
                    data = json.loads(raw_text)
                    print(f"[DEBUG] Parsed from raw_text: {data}")
            except Exception as e:
                print(f"[DEBUG] Failed to parse raw_text: {e}")
                pass

        if not data and request.form:
            data = request.form.to_dict()
            print(f"[DEBUG] Parsed from form: {data}")

        if not data:
            print(f"[ERROR] No data received in request")
            return jsonify({"error": "no_data", "message": "No JSON or form data"}), 400

        # Mapping for short names to long names
        mapping = {
            't': 'temperature',
            'h': 'humidity',
            's': 'soil',
            'r': 'rain',
            'l': 'light'
        }
        for short, long in mapping.items():
            if short in data and long not in data:
                data[long] = data[short]

        # Extract values with defaults
        try:
            t = float(data.get("temperature", 0.0))
            h = float(data.get("humidity", 0.0))
            s = int(data.get("soil", 0))
            r = int(data.get("rain", 4095))
            l = float(data.get("light", 0.0))
            ts = int(data.get("ts") or time.time())
            print(f"[SUCCESS] Parsed values: T={t}, H={h}, S={s}, R={r}, L={l}")
        except (ValueError, TypeError) as e:
            print(f"[ERROR] Invalid data types: {e}")
            return jsonify({"error": "invalid_data_types", "details": str(e)}), 400

        conn = get_db()
        conn.execute(
            "INSERT INTO sensor_readings(ts, temperature, humidity, soil, rain, light) VALUES(?, ?, ?, ?, ?, ?)",
            (ts, t, h, s, r, l)
        )
        conn.commit()
        print(f"[SUCCESS] Data stored in database")

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

        return jsonify({"ok": True, "ts": ts}), 200
    except Exception as e:
        print(f"ERROR in store_sensor_data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/sensors/push", methods=["POST"])
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
        
        conn = get_db()
        conn.execute(
            "INSERT INTO sensor_readings(ts, temperature, humidity, soil, rain, light) VALUES(?, ?, ?, ?, ?, ?)",
            (ts, t, h, s, r, l),
        )
        conn.commit()
        return jsonify({"ok": True})
    except Exception:
        return jsonify({"error": "internal_error"}), 500

@app.route("/sensors/register", methods=["POST"])
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

@app.route("/", methods=["GET"])
def home():
    return send_from_directory('static', 'index.html')

@app.route("/health", methods=["GET"])
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

@app.route("/status", methods=["GET"])
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

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok", "time": int(time.time()), "ok": True})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        json_data = request.get_json(silent=True) or {}
        form_data = request.form or {}
        
        api_key = form_data.get("gemini_api_key") or json_data.get("gemini_api_key") or default_gemini_key()
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
                r = requests.get(image_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                r.raise_for_status()
                content = r.content
            except Exception as e:
                return jsonify({"error": "image_url_fetch_failed", "details": str(e)}), 400
        
        if not content:
            return jsonify({"error": "no_content"}), 400
        
        # Use Gemini for classification
        res = classify_image_with_gemini(content, api_key, language)
        
        if res is None or "error" in res:
            error_details = res.get("error") if res else "Unknown classification error"
            return jsonify({
                "error": "classification_failed", 
                "details": error_details,
                "is_plant": False,
                "guidance": f"Error: {error_details}"
            }), 500
            
        if not res.get("is_plant", True):
            msg = "Not a plant image. Please upload a clear leaf photo."
            if language == "ur":
                msg = "یہ پودے کی تصویر نہیں ہے۔ براہ کرم پتے کی صاف تصویر اپ لوڈ کریں۔"
            return jsonify({
                "is_plant": False,
                "message": res.get("guidance") or msg
            }), 200

        cat_top = res.get("category", "Healthy")
        conf = float(res.get("confidence", 0.9))
        disease_name = res.get("disease_name", cat_top)
        guide = res.get("guidance", "")

        # If we have a file but no image_url, try to upload to Cloudinary so history works
        if not image_url and HAS_CLOUDINARY and content:
            try:
                temp_filename = f"pred_{int(time.time())}_{os.urandom(4).hex()}.jpg"
                temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
                with open(temp_path, "wb") as f:
                    f.write(content)
                c_url, c_err = upload_to_cloudinary(temp_path)
                if c_url:
                    image_url = c_url
                try: os.remove(temp_path)
                except: pass
            except Exception:
                pass

        # Always store the prediction for history
        try:
            store_prediction(image_url or "", disease_name, conf, guide)
        except Exception as e:
            print(f"Failed to store prediction: {e}")
        
        return jsonify({
            "is_plant": True,
            "confidence": conf,
            "top_class_index": 0,
            "top_classes": [{"name": disease_name, "prob": conf}],
            "category_top": cat_top,
            "categories_top3": [{"name": cat_top, "prob": conf}],
            "guidance": guide,
            "image_url": image_url
        })
    except Exception as e:
        print(f"Predict error: {e}")
        return jsonify({"error": "internal_error", "details": str(e)}), 500

@app.route("/predictions", methods=["GET"])
@app.route("/predictions/history", methods=["GET"])
def get_stored_predictions():
    try:
        print("DEBUG: Fetching predictions history")
        limit = request.args.get("limit", 20, type=int)
        predictions = get_predictions(limit=limit)
        return jsonify({"predictions": predictions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
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
            default_msg = "Please ask your local agricultural extension."
            if req_language == "ur":
                default_msg = "براہ کرم اپنے مقامی زرعی توسیعی مرکز سے پوچھیں۔"
            
            return jsonify({"reply": default_msg}), 200
        
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
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

@app.route("/translate", methods=["POST"])
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
            model = genai.GenerativeModel('gemini-2.5-flash')
            lang = "Urdu" if target == "ur" else "English"
            resp = model.generate_content(f"Translate to {lang}. Only translation:\n\n{text}")
            txt = getattr(resp, 'text', None) or (resp.candidates[0].content.parts[0].text if getattr(resp, 'candidates', None) else None)
            return jsonify({"translated": txt or text})
        except Exception:
            return jsonify({"translated": text})
    except Exception:
        return jsonify({"error": "internal_error"}), 500

@app.route("/camera/register", methods=["POST"])
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

@app.route("/camera/latest", methods=["GET"])
def camera_latest():
    """Serves the latest ESP32-CAM image, ensuring consistency across all server workers."""
    try:
        cam_url = get_config("esp32_camera_url", "")
        # Always fetch from DB to ensure workers are in sync
        latest_img = str(get_config("permanent_latest_cam_url", "") or "").strip()
        
        if not latest_img:
            # Fallback 1: Generic latest config
            latest_img = str(get_config("latest_image_url", "") or "").strip()
            
        if not latest_img:
             # Fallback 2: Check Cloudinary API directly (if DB is empty/wiped)
             print("DEBUG: No DB image, checking Cloudinary API...")
             latest_img = latest_cloudinary_image_url()
             
        if latest_img:
             # If we found it in Cloudinary but not DB, restore it to DB for next time
             set_config("permanent_latest_cam_url", latest_img)
             set_config("latest_image_url", latest_img)

        return jsonify({
            "stream_url": cam_url,
            "latest_image_url": latest_img,
            "image_url": latest_img
        }), 200
    except Exception as e:
        print(f"Error in camera_latest: {e}")
        return jsonify({"error": "internal_error"}), 500

@app.route("/camera/upload", methods=["POST"])
def camera_upload():
    try:
        print(f"DEBUG: /camera/upload received. Content-Type: {request.content_type}")
        if not HAS_CLOUDINARY:
            return jsonify({"error": "cloudinary_not_configured"}), 501
            
        filepath = None
        
        # 1. Handle raw binary upload (ESP32 style)
        if request.content_type and ('image/jpeg' in request.content_type or 'image/jpg' in request.content_type):
            data = request.get_data()
            if not data:
                return jsonify({"error": "no_data"}), 400
            
            filename = f"cam_{int(time.time())}.jpg"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            with open(filepath, "wb") as f:
                f.write(data)
                
        # 2. Handle multipart form upload (Standard style)
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "no_filename"}), 400
            filename = f"cam_{int(time.time())}.jpg"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
        else:
            print(f"DEBUG: Unsupported Media Type: {request.content_type}")
            # Fallback: Try reading raw data anyway if content length > 0
            if request.content_length and request.content_length > 0:
                 data = request.get_data()
                 filename = f"cam_{int(time.time())}.jpg"
                 filepath = os.path.join(UPLOAD_FOLDER, filename)
                 with open(filepath, "wb") as f:
                     f.write(data)
            else:
                 return jsonify({"error": "unsupported_media_type", "type": str(request.content_type)}), 415

        cleanup_old_cloud_images()

        # 3. Upload to Cloudinary
        # If Cloudinary is disabled or fails, we might get None
        upload_result = upload_to_cloudinary(filepath)
        
        # Robustly handle the tuple return (url, error) or potentially old single value
        if isinstance(upload_result, tuple):
             url, error_msg = upload_result
        else:
             # Fallback if someone reverts the function signature
             url = upload_result
             error_msg = "Unknown error"
             
        # Debug output
        print(f"DEBUG: upload_result type: {type(upload_result)}")
        print(f"DEBUG: upload_result value: {upload_result}")
        
        # 4. Cleanup local file
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
            
        if not url:
            print(f"DEBUG: Cloud upload failed: {error_msg}")
            return jsonify({"error": "cloud_upload_failed", "details": error_msg}), 500
            
        # 5. Update In-Memory Cache (using new config system)
        set_config("permanent_latest_cam_url", url)
        set_config("latest_image_url", url)
        print(f"DEBUG: Persistent camera URL updated: {url}")
            
        return jsonify({"ok": True, "url": url}), 200
        
    except Exception as e:
        print(f"Error in camera_upload: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload_alias():
    return camera_upload()

@app.route("/latest", methods=["GET"])
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

@app.route("/sensors/history", methods=["GET"])
def sensors_history_endpoint():
    try:
        limit = int(request.args.get("limit", 50))
        conn = get_db()
        cur = conn.execute(
            "SELECT ts, temperature, humidity, soil, rain, light FROM sensor_readings ORDER BY ts DESC LIMIT ?",
            (limit,)
        )
        rows = cur.fetchall()
        results = []
        for r in rows:
            results.append({
                "ts": int(r['ts']),
                "temperature": float(r['temperature'] or 0.0),
                "humidity": float(r['humidity'] or 0.0),
                "soil": int(r['soil'] or 0),
                "rain": int(r['rain'] if r['rain'] is not None else 4095),
                "light": float(r['light'] if len(r) > 5 else 0.0)
            })
        return jsonify({"readings": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predictions/history", methods=["GET"])
def predictions_history():
    try:
        limit = int(request.args.get("limit", 20))
        predictions = get_predictions(limit)
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/bot", methods=["POST"])
def bot():
    """Twilio Webhook to handle incoming WhatsApp messages"""
    if not HAS_TWILIO:
        return "Twilio library not found", 500

    incoming_msg = request.values.get('Body', '').lower()
    resp = MessagingResponse()
    msg = resp.message()

    # Check if user wants readings
    if 'reading' in incoming_msg or 'status' in incoming_msg or 'data' in incoming_msg:
        data = latest_reading()
        if data:
            # Format the reading
            t = data.get('temperature', 0)
            h = data.get('humidity', 0)
            s = data.get('soil', 0)
            r = data.get('rain', 0)
            l = data.get('light', 0)
            ts_val = data.get('ts', time.time())
            ts_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts_val))

            reply_text = (
                f"🌱 *Current Crop Status* 🌱\n"
                f"📅 {ts_str}\n"
                f"🌡 Temp: {t:.1f}°C\n"
                f"💧 Humidity: {h:.1f}%\n"
                f"🌱 Soil: {s}\n"
                f"🌧 Rain: {r}\n"
                f"💡 Light: {l:.1f}"
            )
        else:
            reply_text = "⚠️ No sensor data available yet."
    else:
        reply_text = "🤖 Hello! Send 'reading' to get the latest crop sensor data."

    msg.body(reply_text)
    return str(resp)


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000)
