from preprocess import get_data
from model import build_1d_cnn
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report

# 1. Load data
X_train, X_test, y_train, y_test = get_data("data")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 2. Build model
model = build_1d_cnn(input_shape=(X_train.shape[1], 1))
model.summary()

# 3. Train
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("outputs/best_model.keras", save_best_only=True)
]

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=256,
    validation_split=0.1,
    callbacks=callbacks
)

# 4. Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc:.4f}")

y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))