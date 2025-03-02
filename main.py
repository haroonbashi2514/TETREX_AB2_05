import threading
import time
import logging
import psutil
import os
import subprocess
import asyncio
import joblib
import csv
import hashlib
from datetime import datetime
from pathlib import Path
import platform
import math

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# FastAPI and related imports
from fastapi import FastAPI, Request, WebSocket, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Windows-specific imports (only loaded on Windows)
if platform.system() == "Windows":
    import wmi
    import winreg
    import win32evtlog

# ------------------------------
# Configure Logging
# ------------------------------
# Use WARNING level so debug info isn't printed to console.
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------
# Global Variables
# ------------------------------
file_monitor_handler_global = None
ml_model = None  # Will hold the loaded ML model
response_triggered_flag = threading.Event()  # For thread-safe flag handling
connected_clients = []  # List of active WebSocket clients
main_loop = None  # Global event loop (set during startup)

# ------------------------------
# Model Training and Loading
# ------------------------------
def train_and_save_model():
    data_path = r"C:\Users\Home\jose\data_file.csv"  # Update with your dataset path
    data = pd.read_csv(data_path)
    logging.info(f"Dataset Loaded with shape: {data.shape}")
    if "FileName" in data.columns and "md5Hash" in data.columns:
        data = data.drop(columns=["FileName", "md5Hash"])
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                          random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Model Accuracy: {acc*100:.2f}%")
    logging.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
    joblib.dump(model, "ransomware_model_5features.pkl")
    logging.info("Model saved as 'ransomware_model_5features.pkl'")

def load_ml_model(model_path="ransomware_model_5features.pkl"):
    try:
        model = joblib.load(model_path)
        logging.info("ML model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading ML model: {e}")
        return None

def predict_ransomware(model, features):
    features_array = np.array([features])
    prediction = model.predict(features_array)[0]
    return prediction

# ------------------------------
# System Metrics Functions
# ------------------------------
def get_cpu_usage():
    return psutil.cpu_percent(interval=1)

def get_memory_usage():
    return psutil.virtual_memory().percent

def get_disk_usage():
    return psutil.disk_usage('/').percent

def get_registry_alerts_count():
    count = 0
    if platform.system() == "Windows":
        keys_to_monitor = [r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run"]
        for key in keys_to_monitor:
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key, 0, winreg.KEY_READ) as reg:
                    num = winreg.QueryInfoKey(reg)[0]
                    for i in range(num):
                        name, value, _ = winreg.EnumValue(reg, i)
                        if "ransom" in name.lower() or "encrypt" in name.lower():
                            count += 1
            except Exception:
                continue
    return count

def get_unauthorized_process_count():
    count = 0
    for proc in psutil.process_iter(['name']):
        try:
            name = proc.info.get('name', '').lower()
            if name in ["powershell.exe", "cmd.exe", "wmic.exe"]:
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return count

def get_shadow_copy_flag():
    try:
        output = subprocess.check_output('vssadmin list shadows', shell=True, stderr=subprocess.STDOUT)
        if b"Error" in output:
            return 1
    except subprocess.CalledProcessError:
        return 1
    return 0

def get_suspicious_network_count():
    count = 0
    for conn in psutil.net_connections(kind='inet'):
        if conn.raddr and conn.raddr[0] == "192.168.1.100":  # Adjust as needed.
            count += 1
    return count

def get_total_network_connections():
    return len(psutil.net_connections(kind='inet'))

def get_suspicious_file_extension_count(directory):
    count = 0
    path = Path(directory)
    for file in path.rglob('*'):
        if file.suffix.lower() in {'.locked', '.encrypted'}:
            count += 1
    return count

def compute_file_hash(file_path):
    try:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        logging.error(f"Error computing hash for {file_path}: {e}")
        return None

def analyze_directory_entropy(directory):
    suspicious_count = 0
    path = Path(directory)
    for file_path in path.rglob('*'):
        if file_path.suffix.lower() in ['.docx', '.pdf', '.txt', '.exe']:
            try:
                data = file_path.read_bytes()
                if not data:
                    continue
                probabilities = [data.count(byte) / len(data) for byte in set(data)]
                entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
                if entropy > 7.5:
                    suspicious_count += 1
            except Exception:
                continue
    return suspicious_count

# ------------------------------
# File Monitoring using Watchdog
# ------------------------------
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class FileMonitorHandler(FileSystemEventHandler):
    def _init_(self):
        self.modified_files = set()
        self.renamed_files = set()
        self.deleted_files = set()

    def on_modified(self, event):
        if not event.is_directory:
            self.modified_files.add(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self.modified_files.add(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            self.deleted_files.add(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self.renamed_files.add(event.dest_path)

    def get_file_event_counts(self):
        counts = {
            "modified": len(self.modified_files),
            "renamed": len(self.renamed_files),
            "deleted": len(self.deleted_files)
        }
        self.modified_files.clear()
        self.renamed_files.clear()
        self.deleted_files.clear()
        return counts

def start_file_monitor(path, handler):
    if not path.exists():
        logging.error(f"Path {path} does not exist.")
        return
    observer = Observer()
    observer.schedule(handler, str(path), recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# ------------------------------
# Advanced Monitoring (Placeholders)
# ------------------------------
def monitor_process_execution():
    while True:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                name = proc.info.get('name', '').lower()
                if name in ["powershell.exe", "cmd.exe", "wmic.exe"]:
                    pass  # Placeholder
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        time.sleep(5)

def monitor_shadow_copy_deletion():
    while True:
        if get_shadow_copy_flag():
            pass  # Placeholder
        time.sleep(60)

def monitor_registry_changes():
    while True:
        _ = get_registry_alerts_count()
        time.sleep(5)

def monitor_network_traffic():
    while True:
        _ = get_suspicious_network_count()
        time.sleep(5)

def monitor_memory_usage(memory_threshold=80):
    while True:
        mem_usage = psutil.virtual_memory().percent
        if mem_usage > memory_threshold:
            pass  # Placeholder
        time.sleep(2)

def monitor_file_extension_changes(path):
    while True:
        _ = get_suspicious_file_extension_count(path)
        time.sleep(5)

def monitor_abnormal_system_calls():
    while True:
        time.sleep(5)

def monitor_process_injection():
    while True:
        time.sleep(5)

def monitor_system_calls_etw():
    while True:
        time.sleep(5)

def monitor_user_login_events():
    while True:
        time.sleep(30)

def monitor_advanced_registry_changes():
    while True:
        time.sleep(5)

def monitor_network_flow_details():
    while True:
        _ = get_total_network_connections()
        time.sleep(5)

def start_falco_monitor():
    if platform.system() != "Linux":
        return
    try:
        asyncio.set_event_loop(asyncio.new_event_loop())
        falco_cmd = ["falco", "-o", "json_output=true"]
        process = subprocess.Popen(falco_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while True:
            line = process.stdout.readline()
            if not line:
                break
    except FileNotFoundError:
        logging.error("Falco executable not found in PATH.")
    except Exception as e:
        logging.error(f"Error running Falco: {e}")

# ------------------------------
# WebSocket Data Pushing Function
# ------------------------------
def notify_system_data(data):
    message = {"type": "system_data", "data": data}
    for client in connected_clients:
        try:
            asyncio.run_coroutine_threadsafe(client.send_json(message), main_loop)
        except Exception as e:
            logging.error(f"Error sending system data to client: {e}")

# ------------------------------
# Automated Response Mechanism
# ------------------------------
def reset_response_flag(delay=60):
    time.sleep(delay)
    response_triggered_flag.clear()

def notify_clients(alert_message):
    for client in connected_clients:
        try:
            asyncio.run_coroutine_threadsafe(client.send_json(alert_message), main_loop)
        except Exception as e:
            logging.error(f"Error sending alert to client: {e}")

def trigger_response(data):
    if not response_triggered_flag.is_set():
        response_triggered_flag.set()
        alert_message = {
            "type": "alert",
            "alert": "Potential ransomware activity detected!",
            "data": data
        }
        notify_clients(alert_message)
        threading.Thread(target=reset_response_flag, args=(60,), daemon=True).start()

# ------------------------------
# Data Collection and ML Prediction
# ------------------------------
def collect_system_data():
    file_events = file_monitor_handler_global.get_file_event_counts() if file_monitor_handler_global else {"modified": 0, "renamed": 0, "deleted": 0}
    data = {
        "timestamp": datetime.now().isoformat(),
        "cpu_usage": get_cpu_usage(),
        "memory_usage": get_memory_usage(),
        "disk_usage": get_disk_usage(),
        "modified": file_events.get("modified", 0),
        "renamed": file_events.get("renamed", 0),
        "deleted": file_events.get("deleted", 0),
        "entropy_alerts": analyze_directory_entropy(r"C:\Users\Home\Documents"),
        "unauth_proc_count": get_unauthorized_process_count(),
        "shadow_copy_flag": get_shadow_copy_flag(),
        "registry_alerts_count": get_registry_alerts_count(),
        "susp_net_count": get_suspicious_network_count(),
        "susp_ext_count": get_suspicious_file_extension_count(r"C:\Users\Home\Documents"),
        "proc_injection": 0,      # Placeholder
        "sys_call_anomaly": 0,    # Placeholder
        "total_net_connections": get_total_network_connections()
    }
    return data

def collect_and_predict(model):
    data = collect_system_data()
    features = [
        data["cpu_usage"],
        data["memory_usage"],
        data["disk_usage"],
        data["modified"],
        data["renamed"],
        data["deleted"],
        data["entropy_alerts"],
        data["unauth_proc_count"],
        data["shadow_copy_flag"],
        data["registry_alerts_count"],
        data["susp_net_count"],
        data["susp_ext_count"],
        data["proc_injection"],
        data["sys_call_anomaly"],
        data["total_net_connections"]
    ]
    ml_prediction = predict_ransomware(model, features) if model is not None else 0

    thresholds = {
        "cpu_usage": 80,
        "memory_usage": 80,
        "disk_usage": 90,
        "modified": 10,
        "renamed": 10,
        "deleted": 5,
        "entropy_alerts": 10,
        "unauth_proc_count": 1,
        "shadow_copy_flag": 0,
        "registry_alerts_count": 0,
        "susp_net_count": 0,
        "susp_ext_count": 0,
        "proc_injection": 0,
        "sys_call_anomaly": 0,
        "total_net_connections": 100
    }

    threshold_breach = any(data.get(key, 0) > threshold for key, threshold in thresholds.items())
    if ml_prediction == 1 or threshold_breach:
        ml_prediction = 1
        data["ml_detection"] = True
    else:
        data["ml_detection"] = False

    data["ml_prediction"] = ml_prediction
    if ml_prediction == 1:
        trigger_response(data)
        data["response_triggered"] = True
    else:
        data["response_triggered"] = False

    data["features"] = {
        "cpu_usage": data["cpu_usage"],
        "memory_usage": data["memory_usage"],
        "disk_usage": data["disk_usage"],
        "modified": data["modified"],
        "renamed": data["renamed"],
        "deleted": data["deleted"],
        "entropy_alerts": data["entropy_alerts"],
        "unauth_proc_count": data["unauth_proc_count"],
        "shadow_copy_flag": data["shadow_copy_flag"],
        "registry_alerts_count": data["registry_alerts_count"],
        "susp_net_count": data["susp_net_count"],
        "susp_ext_count": data["susp_ext_count"],
        "proc_injection": data["proc_injection"],
        "sys_call_anomaly": data["sys_call_anomaly"],
        "total_net_connections": data["total_net_connections"]
    }
    return data

def correlation_engine(data, csv_file="correlation_log.csv"):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "cpu_usage", "memory_usage", "disk_usage",
                             "modified", "renamed", "deleted", "entropy_alerts",
                             "unauth_proc_count", "shadow_copy_flag", "registry_alerts_count",
                             "susp_net_count", "susp_ext_count", "proc_injection", "sys_call_anomaly",
                             "total_net_connections", "ml_prediction", "ml_detection", "response_triggered"])
        writer.writerow([
            data.get("timestamp"),
            data.get("cpu_usage"),
            data.get("memory_usage"),
            data.get("disk_usage"),
            data.get("modified"),
            data.get("renamed"),
            data.get("deleted"),
            data.get("entropy_alerts"),
            data.get("unauth_proc_count"),
            data.get("shadow_copy_flag"),
            data.get("registry_alerts_count"),
            data.get("susp_net_count"),
            data.get("susp_ext_count"),
            data.get("proc_injection"),
            data.get("sys_call_anomaly"),
            data.get("total_net_connections"),
            data.get("ml_prediction"),
            data.get("ml_detection"),
            data.get("response_triggered")
        ])

# ------------------------------
# Periodic Display using asyncio
# ------------------------------
async def periodic_display():
    global ml_model
    while True:
        data = collect_and_predict(ml_model)
        correlation_engine(data)
        notify_system_data(data)
        await asyncio.sleep(5)

# ------------------------------
# FastAPI Application Setup
# ------------------------------
app = FastAPI(
    title="Advanced Ransomware Monitoring API",
    description="Tracks 15 system monitoring features with real-time ML detection updates via WebSocket.",
    version="2.0"
)
templates = Jinja2Templates(directory="joo")
app.mount("/static", StaticFiles(directory="joo"), name="static")

@app.websocket("/ws/alerts")
async def websocket_alert_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keeps connection alive.
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)

@app.on_event("startup")
async def startup_event():
    global file_monitor_handler_global, ml_model, main_loop
    monitor_path = Path(r"C:\Users\Home\Documents")  # Adjust as needed.
    file_monitor_handler_global = FileMonitorHandler()
    threading.Thread(target=start_file_monitor, args=(monitor_path, file_monitor_handler_global), daemon=True).start()
    threading.Thread(target=monitor_process_execution, daemon=True).start()
    threading.Thread(target=monitor_shadow_copy_deletion, daemon=True).start()
    threading.Thread(target=monitor_registry_changes, daemon=True).start()
    threading.Thread(target=monitor_network_traffic, daemon=True).start()
    threading.Thread(target=monitor_memory_usage, args=(80,), daemon=True).start()
    threading.Thread(target=monitor_file_extension_changes, args=(str(monitor_path),), daemon=True).start()
    threading.Thread(target=monitor_abnormal_system_calls, daemon=True).start()
    threading.Thread(target=monitor_process_injection, daemon=True).start()
    threading.Thread(target=monitor_system_calls_etw, daemon=True).start()
    threading.Thread(target=monitor_user_login_events, daemon=True).start()
    threading.Thread(target=monitor_advanced_registry_changes, daemon=True).start()
    threading.Thread(target=monitor_network_flow_details, daemon=True).start()
    threading.Thread(target=start_falco_monitor, daemon=True).start()
    
    if not os.path.exists("ransomware_model_5features.pkl"):
        train_and_save_model()
    ml_model = load_ml_model("ransomware_model_5features.pkl")
    main_loop = asyncio.get_running_loop()
    asyncio.create_task(periodic_display())

@app.get("/", summary="Welcome")
def read_root():
    return {"message": "Welcome to the Advanced Ransomware Monitoring API."}

@app.get("/system_data", summary="Get Current System Data and ML Prediction")
def get_system_data():
    data = collect_and_predict(ml_model)
    return data

@app.get("/simulate_attack", summary="Simulate a Ransomware Attack")
def simulate_attack():
    data = collect_system_data()
    data["ml_prediction"] = 1
    data["ml_detection"] = True
    trigger_response(data)
    data["response_triggered"] = True
    return {"message": "Simulated ransomware attack triggered.", "data": data}

@app.get("/dashboard", response_class=HTMLResponse, summary="dashboard")
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/test", summary="Run Self Tests")
def run_tests():
    results = {
        "cpu_usage": get_cpu_usage(),
        "memory_usage": get_memory_usage(),
        "disk_usage": get_disk_usage(),
        "file_monitor_initialized": file_monitor_handler_global is not None
    }
    return results

if _name_ == "_main_":
    uvicorn.run("main:app", host="127.0.0.1", port=8001)