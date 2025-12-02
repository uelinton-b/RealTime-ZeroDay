import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# ==========================
# 1. CARREGAR DADOS
# ==========================
normal = pd.read_csv("normal.csv")
ataque = pd.read_csv("ataque.csv")

# ==========================
# 2. ADICIONAR RÓTULOS
# ==========================
normal["label"] = 0      # 0 = benigno
ataque["label"] = 1      # 1 = ataque

# ==========================
# 3. UNIR DATASETS
# ==========================
df = pd.concat([normal, ataque], ignore_index=True)

print("Total de amostras:", len(df))
print(df["label"].value_counts())

print(df.columns)

# ==========================
# 1. Criar IAT
# ==========================
df = df.sort_values("frame.time_epoch")
df["iat"] = df["frame.time_epoch"].diff().fillna(0)

# Converter tcp.flags de '0x0002' para inteiro
df["tcp.flags"] = df["tcp.flags"].fillna("0x0000")
df["tcp.flags"] = df["tcp.flags"].apply(lambda x: int(x, 16))

df = df.drop(columns=["ip.src", "ip.dst", "frame.time_epoch"])

# ==========================
# 2. Converter IPs em números
# ==========================
le_src = LabelEncoder()
le_dst = LabelEncoder()

#df["ip.src"] = le_src.fit_transform(df["ip.src"])
#df["ip.dst"] = le_dst.fit_transform(df["ip.dst"])

# ==========================
# 3. Selecionar features
# ==========================
features = [
    "iat",
    "frame.len",
    "ip.proto",
    "tcp.flags",
    "tcp.window_size",
    "udp.length",
]

X = df[features]
y = df["label"]

# Divisão
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ==========================
# 4. Random Forest
# ==========================
rf = RandomForestClassifier(
    n_estimators=150, 
    max_depth=12,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# ==========================
# 5. Avaliação
# ==========================
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("Precision (class 1):", precision_score(y_test, y_pred, pos_label=1))
print("Recall (class 1):", recall_score(y_test, y_pred, pos_label=1))
print("F1 (class 1):", f1_score(y_test, y_pred, pos_label=1))
print("\n")
print("\nMatriz de confusão:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# ==========================
# 6. Importância das Features
# ==========================
importances = pd.DataFrame({
    "feature": features,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

print("\nImportâncias:")
print(importances)