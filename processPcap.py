import pyshark
import pandas as pd
import numpy as np
import glob
import gc
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)

# ==============================================================
# 1. Configurações gerais
# ==============================================================

DATASET_DIR = "dataset/"
MODEL_DIR = "modeloTreinado/"
FEATURES = [
    "iat",
    "frame_len",
    "proto",
    "tcp_flags",
    "tcp_win",
    "udp_len",
    "pps_1",
    "bps_1",
    "syn_rate_1",
    "icmp_rate_1",
    "frame_mean_1",
    "pps_5",
    "bps_5",
    "syn_rate_5",
    "icmp_rate_5",
    "frame_mean_5",
]


# ==============================================================
# 2. Função: Extrair features básicas de 1 PCAP → CSV
# ==============================================================

def pcap_to_csv(pcap_file, label, output_csv):
    print(f"\n[+] Processando PCAP: {pcap_file}")

    cap = pyshark.FileCapture(
        pcap_file,
        only_summaries=False,
        use_json=True,
        keep_packets=False
    )

    rows = []
    count = 0

    for pkt in cap:
        try:
            ts = float(pkt.frame_info.time_epoch)
            size = int(pkt.length)

            proto = int(pkt.ip.proto) if hasattr(pkt, "ip") else 0
            tcp_flags = int(pkt.tcp.flags, 16) if hasattr(pkt, "tcp") else 0
            tcp_win = int(pkt.tcp.window_size_value) if hasattr(pkt, "tcp") else 0
            udp_len = int(pkt.udp.length) if hasattr(pkt, "udp") else 0

            sport = int(pkt[pkt.transport_layer].srcport) if hasattr(pkt, "transport_layer") else 0
            dport = int(pkt[pkt.transport_layer].dstport) if hasattr(pkt, "transport_layer") else 0

            dst_ip = pkt.ip.dst if hasattr(pkt, "ip") else "0.0.0.0"

            rows.append([
                ts, size, proto, tcp_flags, tcp_win, udp_len,
                sport, dport, dst_ip, label
            ])

            count += 1

            if count % 10000 == 0:
                print(f"  → {count} pacotes processados...")

        except:
            continue

    cap.close()

    df = pd.DataFrame(rows, columns=[
        "time", "frame_len", "proto", "tcp_flags", "tcp_win", "udp_len",
        "src_port", "dst_port", "dst_ip", "label"
    ])

    df.to_csv(output_csv, index=False)

    print(f"[+] CSV salvo: {output_csv}  ({count} pacotes)")

    del df
    del rows
    gc.collect()


# ==============================================================
# 3. Processar todos PCAPs → gerar CSVs
# ==============================================================

def processar_pcaps():
    pcaps = [
        ("Normal.pcapng", 0),
        ("Normal1.pcapng", 0),
        ("ICMP_flood.pcapng", 1),
        ("SYN-flood.pcapng", 1)
    ]

    for fname, label in pcaps:
        in_file = DATASET_DIR + fname
        out_file = DATASET_DIR + fname.replace(".pcapng", ".csv")
        pcap_to_csv(in_file, label, out_file)


# ==============================================================
# 4. Carregar todos CSVs e montar DataFrame final
# ==============================================================

def carregar_dataset():
    print("\n[+] Carregando CSVs do dataset...")

    csv_files = glob.glob(DATASET_DIR + "*.csv")
    dfs = []

    for f in csv_files:
        print("  → Lendo:", f)
        dfs.append(pd.read_csv(f))

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("time")

    print(f"[+] Dataset final carregado: {len(df)} linhas\n")
    return df


# ==============================================================
# 5. Criar features adicionais (IAT, janelas)
# ==============================================================

def add_window_features(df, window):
    df = df.copy()

    df[f"pps_{window}"] = df["frame_len"].rolling(window, min_periods=1).count()
    df[f"bps_{window}"] = df["frame_len"].rolling(window, min_periods=1).sum()
    df[f"syn_rate_{window}"] = (
        (df["tcp_flags"] == 0x02).astype(int)
         .rolling(window, min_periods=1).sum()
    )
    df[f"icmp_rate_{window}"] = (
        (df["proto"] == 1).astype(int)
         .rolling(window, min_periods=1).sum()
    )
    df[f"frame_mean_{window}"] = df["frame_len"].rolling(window, min_periods=1).mean()

    return df


def gerar_features(df):

    print("[+] Criando feature IAT...")
    df["iat"] = df["time"].diff().fillna(0)

    print("[+] Criando features de janelas...\n")

    df = df.set_index("time")
    df = add_window_features(df, 1)
    df = add_window_features(df, 5)
    df = df.reset_index()

    return df


# ==============================================================
# 6. Treinar o modelo RandomForest
# ==============================================================

def treinar_modelo(df):

    X = df[FEATURES].fillna(0)
    y = df["label"]

    print("[+] Treinando RandomForest...\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    rf.fit(X_train, y_train)

    joblib.dump(rf, MODEL_DIR + "modelo_random_forest.pkl")

    print("[+] Modelo salvo em modeloTreinado/modelo_random_forest.pkl\n")

    y_pred = rf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("\nMatriz de confusão:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    # Importância das features
    imp = pd.DataFrame({
        "feature": FEATURES,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\nIMPORTÂNCIAS:\n", imp)


# ==============================================================
# 7. Main — Pipeline completa
# ==============================================================

if __name__ == "__main__":
    print("\n========== PIPELINE IDS ==========\n")

    processar_pcaps()
    df = carregar_dataset()
    df = gerar_features(df)
    treinar_modelo(df)

    print("\n[✓] FINALIZADO COM SUCESSO!\n")
