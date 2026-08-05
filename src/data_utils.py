"""
Data loading and preprocessing for NSL-KDD.
"""

import os
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


COLS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty_level",
]

ATTACKS = {
    "normal": "normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "mailbomb": "DoS", "apache2": "DoS", "processtable": "DoS", "udpstorm": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L", "phf": "R2L",
    "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L", "snmpgetattack": "R2L", "named": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "sendmail": "R2L", "httptunnel": "R2L", "worm": "R2L", "snmpguess": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "xterm": "U2R", "ps": "U2R", "sqlattack": "U2R",
}


def load_data(data_dir="data"):
    """Download NSL-KDD if needed, return preprocessed splits."""
    os.makedirs(data_dir, exist_ok=True)
    urls = {
        "KDDTrain+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
        "KDDTest+.txt":  "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
    }
    for fname, url in urls.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"  Downloading {fname}...")
            r = requests.get(url, timeout=60); r.raise_for_status()
            open(path, "w").write(r.text)

    train = pd.read_csv(f"{data_dir}/KDDTrain+.txt", names=COLS, header=None).drop("difficulty_level", axis=1)
    test  = pd.read_csv(f"{data_dir}/KDDTest+.txt",  names=COLS, header=None).drop("difficulty_level", axis=1)

    # Encode categoricals
    for col in ["protocol_type", "service", "flag"]:
        le = LabelEncoder()
        le.fit(pd.concat([train[col], test[col]]).unique())
        train[col] = le.transform(train[col])
        test[col]  = le.transform(test[col])

    # Binary labels
    y_train = (train["label"].map(lambda x: ATTACKS.get(x, "unknown")) != "normal").astype(int).values
    y_test  = (test["label"].map(lambda x: ATTACKS.get(x, "unknown"))  != "normal").astype(int).values

    feat_cols = [c for c in train.columns if c != "label"]
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train[feat_cols].values.astype(np.float32))
    X_test  = scaler.transform(test[feat_cols].values.astype(np.float32))

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    print(f"  Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test, feat_cols