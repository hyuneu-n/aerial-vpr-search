import os
import sys

# ==================================================================================
# [DO NOT TOUCH] Jetson Environment & Library Setup
# ==================================================================================
IS_JETSON = os.path.exists('/sys/devices/gpu.0/load')

if IS_JETSON:
    # Force local FAISS build usage on Jetson
    current_dir = os.path.dirname(os.path.abspath(__file__))
    libs_dir = os.path.join(current_dir, 'libs')
    sys.path.insert(0, libs_dir)
    print("🚀 Jetson Environment Detected: Added 'libs/' to sys.path")
    
    # Auto-create __init__.py for faiss package if needed
    faiss_pkg_dir = os.path.join(libs_dir, 'faiss')
    if os.path.exists(faiss_pkg_dir):
        init_file = os.path.join(faiss_pkg_dir, '__init__.py')
        if not os.path.exists(init_file):
            try:
                with open(init_file, 'w') as f:
                    pass # Create empty file
                print(f"🔧 Automatically created {init_file} for package import.")
            except Exception as e:
                print(f"⚠️ Failed to create __init__.py: {e}")

import glob
import json
import shutil
import time
import math
from math import radians, cos, sin, asin, sqrt
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import psutil
import subprocess
import csv
import re
import onnxruntime as ort
import threading

# Configure simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

from models.dino_extractor import DinoExtractor
from models.faiss_index import FaissEngine

# System Logging Configuration
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

fh = logging.FileHandler("debug.log", mode='a', encoding='utf-8')
fh.setFormatter(formatter)
logger.addHandler(fh)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)
logger.addHandler(sh)

def force_log_flush():
    for handler in logger.handlers:
        handler.flush()
    sys.stdout.flush()

# [Global Config]
YAW_FIXED_ALT = None  # yaw-only 모드에서 고도를 강제로 300/600 등으로 고정할 때 사용

# [Hardware Monitors]
def get_gpu_load():
    if IS_JETSON:
        # Jetson Orin 계열에서 확인한 GPU load 경로
        # 값 범위: 0 ~ 1000 → 10으로 나눠서 %
        path = "/sys/devices/platform/bus@0/17000000.gpu/load"
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return float(f.read().strip()) / 10.0
            return 0.0
        except:
            return 0.0
    else:
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=1
            )
            return float(result.stdout.strip())
        except:
            return 0.0

def get_resource_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

class GPUMonitor(threading.Thread):
    def __init__(self, interval=0.1):
        super().__init__()
        self.interval = interval
        self.peak_load = 0.0
        self.stopped = False

    def run(self):
        while not self.stopped:
            load = get_gpu_load()
            if load > self.peak_load:
                self.peak_load = load
            time.sleep(self.interval)

    def stop(self):
        self.stopped = True

class PowerMonitor(threading.Thread):
    def __init__(self, interval=0.1):
        super().__init__()
        self.interval = interval
        self.power_samples = []
        self.stopped = False
        self.process = None

    def run(self):
        try:
            if IS_JETSON:
                self.process = subprocess.Popen(
                    ['tegrastats', '--interval', str(int(self.interval * 1000))],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
                )
                for line in self.process.stdout:
                    if self.stopped: break
                    match = re.search(r'POM_5V_IN\s+(\d+)mW', line)
                    if match:
                        self.power_samples.append(int(match.group(1)) / 1000.0)
            else:
                while not self.stopped:
                    try:
                        result = subprocess.run(
                            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=1
                        )
                        output = result.stdout.strip()
                        if output:
                            self.power_samples.append(float(output))
                    except:
                        pass
                    time.sleep(self.interval)
        except:
            pass

    def stop(self):
        self.stopped = True
        if self.process:
            self.process.terminate()
            self.process.wait()

    def get_average_power(self):
        if not self.power_samples: return 0.0
        return sum(self.power_samples) / len(self.power_samples)

# ==================================================================================
# [Logic] Index Selection & Inference
# ==================================================================================

def select_engines_for_query(q_file, query_meta, engines, filter_strategy):
    """
    filter_strategy:
      0 = No Filter (Use 'main' engine, e.g., Master or Single Index)
      1 = Altitude Filter Only
      2 = Yaw Filter Only
      3 = Dual Filter (Alt + Yaw)
    """

    # [Strategy 0] No Filter (Single Index or Master Index)
    if filter_strategy == 0:
        if "main" in engines:
            return [(engines["main"], "Main_Index")]
        # Fallback if multiple indices loaded but strategy is 0 (shouldn't happen)
        return [(list(engines.values())[0], "Fallback")]

    # For Filtering Strategies (1, 2, 3), we need metadata
    meta = query_meta.get(q_file, {})
    pos = meta.get("position", {})
    att = meta.get("attitude", {})
    
    q_alt_raw = pos.get("altitude", 300.0)
    q_yaw_raw = att.get("yaw", 0.0)

    # --- 1. Altitude Selection ---
    #  - filter_strategy == 2 (Yaw Only) 일 때는 전역 설정값(YAW_FIXED_ALT)을 우선 사용
    #  - 없으면 기본 600m, 나머지 전략은 쿼리 고도 기준으로 300/600 중 가까운 값 선택
    global YAW_FIXED_ALT
    if filter_strategy == 2:  # Yaw Only
        if YAW_FIXED_ALT is not None:
            q_alt = float(YAW_FIXED_ALT)
        else:
            q_alt = 600.0
    else:  # Alt Filter or Dual
        # Simple Nearest Neighbor for Altitude
        q_alt = 600.0 if abs(q_alt_raw - 600) < abs(q_alt_raw - 300) else 300.0

    # --- 2. Yaw Selection ---
    if filter_strategy == 1: # Alt Only
        # Load all yaws for the selected altitude
        # This implies we need to return MULTIPLE engines
        # But for this codebase's split index structure (Flat_300_000), 
        # "Alt Only" effectively means "Search all 8 yaw indices for that altitude"
        candidates_yaw = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
        target_yaws = candidates_yaw # All of them
    else:
        # Dual or Yaw Only: Select Top-2 Closest Yaws
        candidates_yaw = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
        sorted_yaws = sorted(candidates_yaw, key=lambda x: min(abs(x - q_yaw_raw), 360 - abs(x - q_yaw_raw)))
        target_yaws = sorted_yaws[:2]

    # --- 3. Gather Engines ---
    results = []
    for y in target_yaws:
        key = (float(q_alt), float(y)) # Key format: (300.0, 0.0)
        if key in engines:
            results.append((engines[key], f"Idx({int(q_alt)}_{int(y)})"))
    
    return results

def run_inference(q_file, query_dir, query_meta, extractor, engines, ref_data, filter_strategy, k_param=10):
    ref_files, ref_lats, ref_lons, ref_meta = ref_data
    
    start_time = time.perf_counter()
    
    # A. Feature Extraction
    feat = extractor.extract(os.path.join(query_dir, q_file))
    if feat is None: return None

    # B. Index Selection
    selected_engines = select_engines_for_query(q_file, query_meta, engines, filter_strategy)
    
    # C. Search
    all_scores = []
    
    # Search each selected engine
    for engine, key in selected_engines:
        if not engine: continue
        distances, indices = engine.search(feat, k=k_param) # Apply k_param
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx >= 0:
                fname = os.path.basename(engine.ref_labels[idx])
                dist = distances[0][i]
                all_scores.append((dist, fname))
    
    # Sort by distance
    all_scores.sort(key=lambda x: x[0])
    
    # Take Top-5 unique
    faiss_top5 = []
    seen = set()
    for d, fname in all_scores:
        if fname not in seen:
            faiss_top5.append(fname)
            seen.add(fname)
        if len(faiss_top5) >= 10: break # Keep enough for metrics
            
    # Calculate physical distances for Top-5
    faiss_top5_dists_phys = []
    q_pos = query_meta.get(q_file, {}).get("position", {"latitude":0, "longitude":0})
    q_lat, q_lon = radians(q_pos["latitude"]), radians(q_pos["longitude"])

    for fname in faiss_top5:
        if fname in ref_meta:
            r_pos = ref_meta[fname]["position"]
            r_lat, r_lon = radians(r_pos["latitude"]), radians(r_pos["longitude"])
            daa = np.sin((r_lat-q_lat)/2)**2 + np.cos(q_lat)*np.cos(r_lat)*np.sin((r_lon-q_lon)/2)**2
            d_m = 6371000 * 2 * asin(sqrt(daa))
            faiss_top5_dists_phys.append(d_m)
        else:
            faiss_top5_dists_phys.append(-1.0)

    # D. Physical Distance Ground Truth (Geo Top-5)
    valid_ref_keys = set()
    for engine, key in selected_engines:
        if not engine: continue
        for p in engine.ref_labels:
            valid_ref_keys.add(os.path.basename(p))
        
    phys_dists_candidates = []
    q_lat_rad = q_lat
    q_lon_rad = q_lon
    _, _, _, ref_full_meta = ref_data
    
    for ref_name, ref_meta_item in ref_full_meta.items():
        if ref_name not in valid_ref_keys: continue
            
        r_pos = ref_meta_item.get("position", {})
        r_lat_deg = r_pos.get("latitude", 0)
        r_lon_deg = r_pos.get("longitude", 0)
        
        r_lat_rad = math.radians(r_lat_deg)
        r_lon_rad = math.radians(r_lon_deg)
        dlat = r_lat_rad - q_lat_rad
        dlon = r_lon_rad - q_lon_rad
        a = math.sin(dlat/2)**2 + math.cos(q_lat_rad) * math.cos(r_lat_rad) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        dist = 6371000 * c
        phys_dists_candidates.append((dist, ref_name))
    
    phys_dists_candidates.sort(key=lambda x: x[0])
    
    if not phys_dists_candidates:
        phys_top5 = []
        phys_top5_dists = []
    else:
        phys_top5 = [x[1] for x in phys_dists_candidates[:5]]
        phys_top5_dists = [x[0] for x in phys_dists_candidates[:5]]

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0

    is_t1 = 1 if (faiss_top5 and faiss_top5[0] in phys_top5) else 0
    is_t5 = 1 if set(faiss_top5) & set(phys_top5) else 0

    used_indices = "|".join([k for _, k in selected_engines])

    return {
        "q_file": q_file,
        "faiss_top5": faiss_top5,
        "faiss_top5_dists": faiss_top5_dists_phys,
        "phys_top5": phys_top5,
        "phys_top5_dists": phys_top5_dists,
        "t1_hit": is_t1,
        "t5_hit": is_t5,
        "latency": latency_ms,
        "fps": fps,
        "gpu_load": get_gpu_load(),
        "mem_mb": get_resource_usage(),
        "index_name": used_indices
    }

def print_performance_report(results, latencies, power_samples, exp_name):
    if not results: return
    total = len(results)
    t1 = sum(r['t1_hit'] for r in results)
    t5 = sum(r['t5_hit'] for r in results)
    avg_error = sum(r['faiss_top5_dists'][0] for r in results if r['faiss_top5_dists']) / total
    
    avg_latency = sum(latencies) / len(latencies)
    avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0
    avg_power = sum(power_samples) / len(power_samples) if power_samples else 0.0
    eff = avg_fps / avg_power if avg_power > 0 else 0.0
    
    max_gpu = max([r['gpu_load'] for r in results])
    max_mem = max([r['mem_mb'] for r in results])

    print("\n" + "="*70)
    print(f"{'FINAL PERFORMANCE REPORT':^70}")
    print("="*70)
    print(f"Experiment      : {exp_name}")
    print(f"Query Images    : {total}")
    print("-"*70)
    print(f"{'METRICS':^70}")
    print("-"*70)
    print(f"Hit@1 Rate  : {t1/total*100:6.2f}% ({t1}/{total})")
    print(f"Hit@k Rate  : {t5/total*100:6.2f}% ({t5}/{total})")
    print(f"Average Error   : {avg_error:6.2f} meters")
    print("-"*70)
    print(f"{'PERFORMANCE METRICS':^70}")
    print("-"*70)
    print(f"Average Latency : {avg_latency:6.2f} ms")
    print(f"Average Throughput     : {avg_fps:6.2f} images/sec")
    print(f"Peak GPU Load   : {max_gpu:6.1f} %")
    print(f"Peak Memory     : {max_mem:6.2f} MB")
    print(f"Average Power   : {avg_power:6.2f} W")
    print("-"*70)
    print(f"{'EFFICIENCY METRIC':^70}")
    print("-"*70)
    print(f"Throughput/Watt        : {eff:6.2f} (images/sec/W) ⭐")
    print("="*70)

# ==================================================================================
# Main Execution Block
# ==================================================================================
def main():
    print("\n" + "="*60)
    print(" 🧪 UAM Visual Localization Experiment Runner")
    print("="*60)
    print(" [Exp 1.1] Index Type Benchmarking (Base: 600m)")
    print("   1. Flat (Baseline)")
    print("   2. IVF50")
    print("   3. IVF100")
    print("   4. IVF200")
    print("   5. HNSW32")
    print("-" * 30)
    print(" [Exp 1.2] Sampling Density Analysis (Base: Flat)")
    print("   6. 300m Only (Dense)")
    print("   7. 600m Only (Sparse)")
    print("   8. Master Integrated (All Data)")
    print("-" * 30)
    print(" [Exp 1.3] k-Parameter Sensitivity (Base: 600m Flat)")
    print("   9. Custom k value test")
    print("-" * 30)
    print(" [Exp 1.6] Spatial Filtering Strategy")
    print("   10. Master Index (No Filter)")
    print("   11. Altitude Filter Only")
    print("   12. Yaw Filter Only @600m")
    print("   13. Dual Filter (Alt + Yaw)")
    print("   14. Yaw Filter Only @300m")
    print("="*60)

    global YAW_FIXED_ALT
    YAW_FIXED_ALT = None  # 기본값 리셋

    try:
        choice = input("Select Experiment Number (1-13): ").strip()
        mode = int(choice)
    except:
        mode = 1

    # Default Configuration
    ref_dir = "database/ref"
    dino_dim = 384
    engines = {}
    
    # Experiment Config Variables
    load_split_indices = False
    index_file_to_load = None
    filter_strategy = 0 # 0: None, 1: Alt, 2: Yaw, 3: Dual
    k_val = 10
    exp_name = "Unknown"

    # --- Configuration Mapping ---
    if mode == 1:
        exp_name = "1.1_Flat_600m"
        index_file_to_load = "600_Flat.index"
    elif mode == 2:
        exp_name = "1.1_IVF50_600m"
        index_file_to_load = "600_IVF50_Flat.index"
    elif mode == 3:
        exp_name = "1.1_IVF100_600m"
        index_file_to_load = "600_IVF100_Flat.index"
    elif mode == 4:
        exp_name = "1.1_IVF200_600m"
        index_file_to_load = "600_IVF200_Flat.index"
    elif mode == 5:
        exp_name = "1.1_HNSW32_600m"
        index_file_to_load = "600_HNSW32.index"
    
    elif mode == 6:
        exp_name = "1.2_Density_300m"
        index_file_to_load = "300_Flat.index"
    elif mode == 7:
        exp_name = "1.2_Density_600m"
        index_file_to_load = "600_Flat.index"
    elif mode == 8:
        exp_name = "1.2_Density_Master"
        index_file_to_load = "Master_All.index"
        
    elif mode == 9:
        exp_name = "1.3_k_Sensitivity"
        index_file_to_load = "600_Flat.index"
        try:
            k_in = input("Enter k value (1, 3, 5, 10, 20, 50): ").strip()
            k_val = int(k_in)
            exp_name += f"_k{k_val}"
        except: k_val = 10
        
    elif mode == 10:
        exp_name = "1.6_Spatial_NoFilter(Master)"
        index_file_to_load = "Master_All.index"
        filter_strategy = 0
    elif mode == 11:
        exp_name = "1.6_Spatial_AltFilter"
        load_split_indices = True
        filter_strategy = 1
    elif mode == 12:
        exp_name = "1.6_Spatial_YawFilter_600m"
        load_split_indices = True
        filter_strategy = 2
        YAW_FIXED_ALT = 600.0
    elif mode == 13:
        exp_name = "1.6_Spatial_DualFilter"
        load_split_indices = True
        filter_strategy = 3
    elif mode == 14:
        exp_name = "1.6_Spatial_YawFilter_300m"
        load_split_indices = True
        filter_strategy = 2
        YAW_FIXED_ALT = 300.0
    
    else:
        print("Invalid Selection. Defaulting to Flat 600m.")
        exp_name = "Default_Flat"
        index_file_to_load = "600_Flat.index"

    print(f"🚀 Initializing Experiment: {exp_name}")

    # --- Loading Engines ---
    if load_split_indices:
        print("📂 Loading Split Indices (This may take a moment)...")
        # Load all Flat_XXX_YYY indices
        count = 0
        for f in tqdm(os.listdir(ref_dir), desc="Loading Splits"):
            if f.endswith(".index") and "Flat_" in f: 
                # e.g. Flat_300_000.index
                parts = f.replace(".index", "").split("_")
                if len(parts) >= 3:
                    try:
                        alt = float(parts[1])
                        yaw = float(parts[2])
                        engines[(alt, yaw)] = FaissEngine(os.path.join(ref_dir, f), dim=dino_dim)
                        count += 1
                    except: pass
        print(f"✅ Loaded {count} split indices.")
    else:
        # Load Single File
        full_path = os.path.join(ref_dir, index_file_to_load)
        if not os.path.exists(full_path):
            print(f"❌ Error: Index file not found: {full_path}")
            return
        print(f"📂 Loading Index: {index_file_to_load}")
        engines["main"] = FaissEngine(full_path, dim=dino_dim)

    # Load Model
    extractor = DinoExtractor("model/model.onnx")
    print("✅ DINO Extractor Initialized.")

    # Load Query & Meta
    query_dir = "test_images"
    query_files = sorted([f for f in os.listdir(query_dir) if f.lower().endswith(('.jpg', '.png'))])
    
    q_json_files = [f for f in os.listdir(query_dir) if f.endswith('.json')]
    query_meta = {}
    if q_json_files:
        with open(os.path.join(query_dir, q_json_files[0]), 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
            query_meta = {os.path.basename(i["imageref"]): i for i in data.get("images", [])}
            
    # Load Ref Meta (Always Master Meta)
    ref_meta = {}
    with open("database/ref/ref_alt_yaw_master.json", 'r') as f:
        data = json.load(f)
        ref_meta = {os.path.basename(i["imageref"]): i for i in data.get("images", [])}
    
    ref_keys = list(ref_meta.keys())
    ref_lats = np.array([radians(ref_meta[k]["position"]["latitude"]) for k in ref_keys])
    ref_lons = np.array([radians(ref_meta[k]["position"]["longitude"]) for k in ref_keys])
    ref_data = (ref_keys, ref_lats, ref_lons, ref_meta)

    # --- Run Inference ---
    results = []
    latencies = []
    
    gpu_mon = GPUMonitor()
    p_mon = PowerMonitor()
    gpu_mon.start()
    p_mon.start()
    
    try:
        for qf in tqdm(query_files, desc="Processing"):
            res = run_inference(qf, query_dir, query_meta, extractor, engines, ref_data, filter_strategy, k_val)
            if res:
                results.append(res)
                latencies.append(res['latency'])
    finally:
        gpu_mon.stop()
        p_mon.stop()
        gpu_mon.join()
        p_mon.join()

    # --- Save Results ---
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_name = f"output/exp_result_{exp_name}_{ts}.csv"
    os.makedirs("output", exist_ok=True)
    
    with open(csv_name, 'w', newline='', encoding='utf-8') as f:
        cols = ['query', 'match_file', 'match_dist_m', 't1', 't5', 'latency_ms', 'fps', 'gpu', 'mem', 'power', 'index', 'exp_mode', 'k_param']
        for i in range(1, 6):
            cols.extend([f'top{i}_file', f'top{i}_dist'])
            
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        
        avg_p = p_mon.get_average_power()
        
        for r in results:
            row = {
                'query': r['q_file'],
                'match_file': r['faiss_top5'][0] if r['faiss_top5'] else "N/A",
                'match_dist_m': f"{r['faiss_top5_dists'][0]:.2f}" if r['faiss_top5_dists'] else "N/A",
                't1': r['t1_hit'],
                't5': r['t5_hit'],
                'latency_ms': f"{r['latency']:.2f}",
                'fps': f"{r['fps']:.2f}",
                'gpu': r['gpu_load'],
                'mem': f"{r['mem_mb']:.1f}",
                'power': f"{avg_p:.2f}",
                'index': r['index_name'],
                'exp_mode': exp_name,
                'k_param': k_val
            }
            for i in range(5):
                if i < len(r['faiss_top5']):
                    row[f'top{i+1}_file'] = r['faiss_top5'][i]
                    row[f'top{i+1}_dist'] = f"{r['faiss_top5_dists'][i]:.2f}"
                else:
                    row[f'top{i+1}_file'] = "N/A"
                    row[f'top{i+1}_dist'] = "N/A"
            writer.writerow(row)
            
    print(f"\n✅ Saved Results: {csv_name}")
    print_performance_report(results, latencies, p_mon.power_samples, exp_name)

if __name__ == "__main__":
    main()