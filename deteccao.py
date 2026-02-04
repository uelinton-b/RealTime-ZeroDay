import subprocess
import joblib
import re
import time
import numpy as np
from collections import deque
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

# -------------------------------
# Carregar modelo
# -------------------------------
model_dir = "modeloTreinado/"
MODEL_PATH = model_dir + "modelo_random_forest.pkl"
model = joblib.load(MODEL_PATH)

# -------------------------------
# Buffers das janelas
# -------------------------------
win1 = deque()
win5 = deque()
sizes_1 = deque()
sizes_5 = deque()

last_packet_time = time.time()

print("🔍 IDS iniciado — Escutando interface...\n")

# -------------------------------
# Regex
# -------------------------------
ip_regex = r"IP (\d+\.\d+\.\d+\.\d+)\.(\d+) > (\d+\.\d+\.\d+\.\d+)\.(\d+):"
flags_regex = r"Flags \[([A-Z]+)\]"
len_regex = r"length (\d+)"

def extract_packet_info(line):
    proto = 0
    if "TCP" in line: proto = 1
    elif "UDP" in line: proto = 2
    elif "ICMP" in line: proto = 3

    ip_match = re.search(ip_regex, line)
    flags_match = re.search(flags_regex, line)
    len_match = re.search(len_regex, line)
    win_match = re.search(r"win (\d+)", line)

    if not ip_match or not len_match:
        return None

    length = int(len_match.group(1))
    tcp_win = int(win_match.group(1)) if win_match else 0

    tcp_flags_str = flags_match.group(1) if flags_match else ""
    syn = 1 if "S" in tcp_flags_str else 0
    ack = 1 if "A" in tcp_flags_str else 0
    rst = 1 if "R" in tcp_flags_str else 0
    fin = 1 if "F" in tcp_flags_str else 0
    tcp_flags = syn * 8 + ack * 4 + rst * 2 + fin * 1

    udp_len = length if proto == 2 else 0

    return {
        "proto": proto,
        "length": length,
        "syn": syn,
        "icmp": 1 if proto == 3 else 0,
        "tcp_flags": tcp_flags,
        "tcp_win": tcp_win,
        "udp_len": udp_len
    }

def compute_window_features(window, sizes):
    now = time.time()
    while window and now - window[0]["time"] > window[0]["win_size"]:
        window.popleft()

    pps = len(window)
    bps = sum(pkt["size"] for pkt in window)
    syn_rate = sum(pkt["syn"] for pkt in window)
    icmp_rate = sum(pkt["icmp"] for pkt in window)

    frame_mean = sum(sizes) / len(sizes) if sizes else 0
    return pps, bps, syn_rate, icmp_rate, frame_mean


# -------------------------------
# tcpdump com filtro IP
# -------------------------------
cmd = [
    "sudo", "tcpdump", "-i", "enp62s0", "-l", "-n",
    "(host 192.168.1.106 or host 192.168.1.107 or host 192.168.1.108 or host 192.168.1.102)",
    "and (tcp or udp or icmp)"
]
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


# =============================================================
#                     LOOP PRINCIPAL
# =============================================================
attack_count = 0
last_alert_time = 0     # cooldown 2s

while True:
    line = process.stdout.readline()
    if not line:
        continue

    now = time.time()
    info = extract_packet_info(line)
    if info is None:
        continue

    # ==== IAT ====
    iat = now - last_packet_time
    last_packet_time = now

    # ==== Atualizar janelas ====
    win1.append({"time": now, "size": info["length"], "syn": info["syn"], "icmp": info["icmp"], "win_size": 1})
    win5.append({"time": now, "size": info["length"], "syn": info["syn"], "icmp": info["icmp"], "win_size": 5})

    sizes_1.append(info["length"])
    sizes_5.append(info["length"])
    if len(sizes_1) > 1000: sizes_1.popleft()
    if len(sizes_5) > 5000: sizes_5.popleft()

    # ==== Features ====
    pps_1, bps_1, syn_rate_1, icmp_rate_1, frame_mean_1 = compute_window_features(win1, sizes_1)
    pps_5, bps_5, syn_rate_5, icmp_rate_5, frame_mean_5 = compute_window_features(win5, sizes_5)

    X = np.array([
        iat,
        info["length"],
        info["proto"],
        info["tcp_flags"],
        info["tcp_win"],
        info["udp_len"],
        pps_1,
        bps_1,
        syn_rate_1,
        icmp_rate_1,
        frame_mean_1,
        pps_5,
        bps_5,
        syn_rate_5,
        icmp_rate_5,
        frame_mean_5
    ]).reshape(1, -1)

    # ==== Predição ====
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]  # probabilidade classe "ataque"

    # Exibir alerta só se ataque detectado
    if pred == 1:  
        horario = datetime.now().strftime("%H:%M:%S")
        print("\n🚨 ATAQUE DETECTADO!")
        print(f"Probabilidade: {proba:.3f}")
        print(f"Horário: {horario}\n")