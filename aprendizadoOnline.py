import subprocess
import time
import csv
from river import naive_bayes, metrics
import pickle


# =====================
# CONFIGURAÇÃO
# =====================
INTERFACE = "enp1s0"
WINDOW_SIZE = 1.0  # segundos
LOG_FILE = "teste.csv"

# Experimento: 1, 2 ou 3
EXPERIMENT_ID = 3

# =====================
# MODELO E MÉTRICAS
# =====================
model = naive_bayes.GaussianNB()

accuracy = metrics.Accuracy()
precision = metrics.Precision()
recall = metrics.Recall()

# =====================
# JANELA
# =====================
window_packets = 0
window_bytes = 0
window_start = None

# =====================
# TCPDUMP
# =====================
cmd = [
    "tcpdump",
    "-i", INTERFACE,
    "-tt",
    "-n",
    "-l",
    "-v",
    "(",
    "host", "192.168.1.108",
    "or", "host", "192.168.1.109",
    "or", "host", "192.168.1.113",
    "or", "host", "192.168.1.103",
    ")"
]

proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    text=True
)

# =====================
# UTILIDADES
# =====================
def extract_packet_length(line):
    try:
        if "length" in line:
            return int(line.split("length")[-1].strip().split()[0])
    except:
        pass
    return 0

# def get_label():
#     """
#     Rótulo controlado pelo experimento
#     0 = normal
#     1 = ataque
#     """
#     if EXPERIMENT_ID == 1:
#         return 0  # só normal

#     elif EXPERIMENT_ID == 2:
#         return 1  # ataque contínuo

#     elif EXPERIMENT_ID == 3:
#         # ataque intermitente (10s ataque / 10s normal)
#         return 1 if int(time.time()) % 20 < 10 else 0


# =====================
# LOGGER
# =====================
with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp",
        "packets",
        "bytes",
        "pkt_rate",
        "byte_rate",
        "y",
        "y_pred",
        "confidence",
        "accuracy",
        "precision",
        "recall"
    ])

STATE_FILE = "attack_state.txt"

def get_label():
    try:
        with open(STATE_FILE, "r") as f:
            value = f.read().strip()
            return int(value)
    except:
        return 0  # default seguro

# =====================
# LOOP PRINCIPAL
# =====================
for line in proc.stdout:
    now = time.time()

    if window_start is None:
        window_start = now

    window_packets += 1
    window_bytes += extract_packet_length(line)

    if now - window_start >= WINDOW_SIZE:
        duration = now - window_start

        x = {
            "packets": window_packets,
            "bytes": window_bytes,
            "pkt_rate": window_packets / duration,
            "byte_rate": window_bytes / duration
        }

        # ===== PREDIÇÃO =====
        y_pred = model.predict_one(x)
        #print(y_pred)
        proba = model.predict_proba_one(x)
        #print(proba)
        conf = max(proba.values()) if proba else 0.0

        # ===== RÓTULO =====
        y = get_label()

        # ===== AVALIAÇÃO (ANTES DE APRENDER) =====
        if y_pred is not None:
            accuracy.update(y, y_pred)
            precision.update(y, y_pred)
            recall.update(y, y_pred)

        # ===== APRENDIZADO ONLINE =====
        model.learn_one(x, y)

        # ===== LOG =====
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                now,
                x["packets"],
                x["bytes"],
                x["pkt_rate"],
                x["byte_rate"],
                y,
                y_pred,
                conf,
                accuracy.get(),
                precision.get(),
                recall.get()
            ])

        print(
            f"[EXP {EXPERIMENT_ID}] "
            f"x={x} | y={y} | pred={y_pred} | "
            f"conf={conf:.2f} | acc={accuracy.get():.3f}"
        )

        # Reset janela
        window_packets = 0
        window_bytes = 0
        window_start = now


with open("modelo_nb.pkl", "wb") as f:
    pickle.dump(model, f)