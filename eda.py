from preprocess import load_raw
from data_profiling import ProfileReport

train_df, test_df = load_raw("data")

profile = ProfileReport(train_df, title="NSL-KDD EDA")
profile.to_file("outputs/nslkdd_eda.html")
print("EDA report saved to outputs/nslkdd_eda.html")