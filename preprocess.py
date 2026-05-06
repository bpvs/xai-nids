import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

cols = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

def load_raw(data_dir):
    """Just load the raw dataframes — used by eda.py."""
    train_df = pd.read_csv(f"{data_dir}/KDDTrain+.txt", header=None, names=cols)
    test_df  = pd.read_csv(f"{data_dir}/KDDTest+.txt",  header=None, names=cols)
    train_df.drop("difficulty", axis=1, inplace=True)
    test_df.drop("difficulty",  axis=1, inplace=True)
    return train_df, test_df

def get_data(data_dir):
    """Full pipeline: load + preprocess + reshape for 1D-CNN."""
    train_df, test_df = load_raw(data_dir)

    # One-hot encode categoricals
    cat_cols = ["protocol_type", "service", "flag"]
    train_df = pd.get_dummies(train_df, columns=cat_cols)
    test_df  = pd.get_dummies(test_df,  columns=cat_cols)
    train_df, test_df = train_df.align(test_df, join="left", axis=1, fill_value=0)

    # Binary labels
    train_df["label"] = train_df["label"].apply(lambda x: 0 if x == "normal" else 1)
    test_df["label"]  = test_df["label"].apply(lambda x: 0 if x == "normal" else 1)

    # Split
    X_train = train_df.drop("label", axis=1).values.astype(np.float32)
    y_train = train_df["label"].values
    X_test  = test_df.drop("label", axis=1).values.astype(np.float32)
    y_test  = test_df["label"].values

    # Scale
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Reshape for 1D-CNN
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)

    return X_train, X_test, y_train, y_test