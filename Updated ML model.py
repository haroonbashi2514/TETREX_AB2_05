import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization, Bidirectional, GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import ADASYN
import random
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# =============================================================================
# 1. Data Distribution Diagnostics
# =============================================================================
def check_data_distribution(X, feature_names):
    print("=== Data Distribution Statistics ===")
    for i, fname in enumerate(feature_names):
        col = X[:, i]
        print(f"{fname}: min={np.min(col):.4f}, max={np.max(col):.4f}, mean={np.mean(col):.4f}, std={np.std(col):.4f}, "
              f"1st percentile={np.percentile(col,1):.4f}, 99th percentile={np.percentile(col,99):.4f}")

# =============================================================================
# 2. Data Ingestion and Exploration
# =============================================================================
def load_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset file not found! Ensure you have downloaded Obfuscated-MalMem2022.")
    df = pd.read_parquet(dataset_path)
    print("✅ Dataset Loaded! Shape:", df.shape)
    print("Dataset Info:")
    print(df.info())
    print("Missing values per column:\n", df.isnull().sum())
    return df

# =============================================================================
# 3. Data Preprocessing and Label Conversion
# =============================================================================
def preprocess_data(df, drop_columns=["FileName", "Timestamp"]):
    # Drop unwanted columns
    df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors='ignore')
    # Identify label column ("Category" or "Class")
    label_column = "Category" if "Category" in df.columns else "Class"
    # Convert labels to binary: benign=0, malware/ransomware=1
    df['binary_label'] = df[label_column].apply(lambda x: 0 if 'benign' in str(x).lower() else 1)
    df = df.drop(columns=[label_column])
    # One-hot encode any remaining categorical features
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_features:
        df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    # Get feature names for diagnostics
    feature_names = df.drop(columns=['binary_label']).columns.tolist()
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop(columns=['binary_label']))
    y = df['binary_label']
    print("Class Distribution:\n", pd.Series(y).value_counts())
    return X, y, scaler, categorical_features, feature_names

# =============================================================================
# 4. Split and Data Augmentation
# =============================================================================
def split_and_augment(X, y, test_size=0.2, random_state=42):
    # Split data (test set remains with original distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    # Use ADASYN to augment the training data
    adasyn = ADASYN(random_state=random_state)
    X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)
    print("After ADASYN, training class distribution:\n", pd.Series(y_train_res).value_counts())
    return X_train_res, X_test, y_train_res, y_test

# =============================================================================
# 5. Enhanced Debug Callback for Autoencoder
# =============================================================================
class DebugCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, sample_size=5):
        super(DebugCallback, self).__init__()
        self.validation_data = validation_data
        self.sample_size = sample_size

    def on_epoch_end(self, epoch, logs=None):
        val_x, _ = self.validation_data
        pred = self.model.predict(val_x, verbose=0)
        custom_val_loss = np.mean(np.square(val_x - pred))
        print(f"🔍 Epoch {epoch+1}: Custom computed val_loss: {custom_val_loss:.6f}")
        print("🔍 Prediction stats on validation set: min =", np.min(pred),
              "max =", np.max(pred), "mean =", np.mean(pred))
        sample_indices = np.random.choice(val_x.shape[0], self.sample_size, replace=False)
        for i in sample_indices:
            diff = np.abs(val_x[i] - pred[i])
            print(f"🔍 Sample {i}: diff mean = {np.mean(diff):.6f}, diff max = {np.max(diff):.6f}")

# =============================================================================
# 6. Feature Extraction: Denoising Autoencoder with Regularization
# =============================================================================
def train_autoencoder(X_train, X_val, epochs=50, batch_size=64, denoising=False, noise_std=0.1):
    input_dim = X_train.shape[1]
    input_layer = Input(shape=(input_dim,))
    # If denoising, add noise to inputs
    if denoising:
        noisy_input = GaussianNoise(noise_std)(input_layer)
        x = noisy_input
    else:
        x = input_layer
    # Encoder with L2 regularization
    encoded = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(encoded)
    # Decoder with L2 regularization
    decoded = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(encoded)
    decoded = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    debug_cb = DebugCallback(validation_data=(X_val, X_val))
    autoencoder.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(X_val, X_val),
        callbacks=[early_stop, lr_reduce, debug_cb]
    )
    encoder_model = Model(inputs=input_layer, outputs=encoded)
    return encoder_model

# =============================================================================
# 7. Base Model Training: XGBoost and RandomForest with Hyperparameter Tuning
# =============================================================================
def train_base_models(X_train_enc, y_train, cv_folds=3):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    # XGBoost
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    param_grid_xgb = {'n_estimators': [100, 200],
                      'max_depth': [5, 7],
                      'learning_rate': [0.05, 0.1]}
    grid_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=cv, scoring='f1', n_jobs=-1)
    grid_xgb.fit(X_train_enc, y_train)
    best_xgb = grid_xgb.best_estimator_
    print("Best XGBoost Params:", grid_xgb.best_params_)
    # RandomForest
    rf_model = RandomForestClassifier(random_state=42)
    param_grid_rf = {'n_estimators': [100, 200],
                     'max_depth': [None, 10, 20]}
    grid_rf = GridSearchCV(rf_model, param_grid_rf, cv=cv, scoring='f1', n_jobs=-1)
    grid_rf.fit(X_train_enc, y_train)
    best_rf = grid_rf.best_estimator_
    print("Best RandomForest Params:", grid_rf.best_params_)
    return best_xgb, best_rf

# =============================================================================
# 8. LSTM Training for Behavioral Analysis (with Bidirectional LSTM)
# =============================================================================
def train_lstm_model(X_train_enc, y_train, X_val_enc, y_val, epochs=30, batch_size=64):
    X_train_lstm = X_train_enc.reshape(X_train_enc.shape[0], 1, X_train_enc.shape[1])
    X_val_lstm = X_val_enc.reshape(X_val_enc.shape[0], 1, X_val_enc.shape[1])
    lstm_input = Input(shape=(1, X_train_enc.shape[1]))
    lstm_layer = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(lstm_input)
    lstm_layer = Bidirectional(LSTM(64, dropout=0.2))(lstm_layer)
    lstm_layer = Dense(32, activation='relu')(lstm_layer)
    lstm_layer = Dropout(0.3)(lstm_layer)
    lstm_output = Dense(1, activation='sigmoid')(lstm_layer)
    lstm_model = Model(inputs=lstm_input, outputs=lstm_output)
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstm_model.fit(
        X_train_lstm, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_lstm, y_val),
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    )
    return lstm_model

# =============================================================================
# 9. Stacking Ensemble: Train Meta-Model
# =============================================================================
def train_stacking_ensemble(X_train_enc, X_train_enc_lstm, y_train, best_xgb, best_rf, lstm_model):
    p_xgb_train = best_xgb.predict_proba(X_train_enc)[:, 1].reshape(-1, 1)
    p_rf_train = best_rf.predict_proba(X_train_enc)[:, 1].reshape(-1, 1)
    p_lstm_train = lstm_model.predict(X_train_enc_lstm).ravel().reshape(-1, 1)
    meta_features_train = np.hstack([p_xgb_train, p_rf_train, p_lstm_train])
    meta_model = LogisticRegression()
    meta_model.fit(meta_features_train, y_train)
    return meta_model

# =============================================================================
# 10. Ensemble Prediction Function (Stacking)
# =============================================================================
def ensemble_predict(X_enc, X_enc_lstm, best_xgb, best_rf, lstm_model, meta_model, threshold=0.5):
    p_xgb = best_xgb.predict_proba(X_enc)[:, 1].reshape(-1, 1)
    p_rf = best_rf.predict_proba(X_enc)[:, 1].reshape(-1, 1)
    p_lstm = lstm_model.predict(X_enc_lstm).ravel().reshape(-1, 1)
    meta_features = np.hstack([p_xgb, p_rf, p_lstm])
    ensemble_prob = meta_model.predict_proba(meta_features)[:, 1]
    ensemble_pred = (ensemble_prob >= threshold).astype(int)
    return ensemble_pred, ensemble_prob

# =============================================================================
# 11. Model Saving
# =============================================================================
def save_models(best_xgb, best_rf, meta_model, lstm_model, encoder, scaler):
    joblib.dump(best_xgb, "ransomware_xgb.pkl")
    joblib.dump(best_rf, "ransomware_rf.pkl")
    joblib.dump(meta_model, "stacking_meta_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    lstm_model.save("ransomware_lstm.h5")
    joblib.dump(encoder, "encoder.pkl")
    print("✅ Models and preprocessors saved.")

# =============================================================================
# 12. Real-Time File Monitoring & Detection
# =============================================================================
class FileMonitorHandler(FileSystemEventHandler):
    def __init__(self, scaler, encoder, categorical_features, best_xgb, best_rf, lstm_model, meta_model, drop_columns):
        self.scaler = scaler
        self.encoder = encoder
        self.categorical_features = categorical_features
        self.best_xgb = best_xgb
        self.best_rf = best_rf
        self.lstm_model = lstm_model
        self.meta_model = meta_model
        self.drop_columns = drop_columns

    def on_modified(self, event):
        if event.is_directory:
            return
        print(f"⚠️ File Modified: {event.src_path}")
        scan_new_file(event.src_path, self.scaler, self.encoder,
                      self.categorical_features, self.best_xgb, self.best_rf,
                      self.lstm_model, self.meta_model, self.drop_columns)

def scan_new_file(file_path, scaler, encoder, categorical_features, best_xgb, best_rf, lstm_model, meta_model, drop_columns):
    try:
        file_data = pd.read_parquet(file_path)
        file_data = file_data.drop(columns=[col for col in drop_columns if col in file_data.columns], errors='ignore')
        if categorical_features:
            file_data = pd.get_dummies(file_data, columns=categorical_features, drop_first=True)
        file_data = scaler.transform(file_data)
        file_encoded = encoder.predict(file_data)
        file_encoded_lstm = file_encoded.reshape(file_encoded.shape[0], 1, file_encoded.shape[1])
        pred_ensemble, _ = ensemble_predict(file_encoded, file_encoded_lstm, best_xgb, best_rf, lstm_model, meta_model, threshold=0.5)
        final_prediction = "Ransomware" if np.any(pred_ensemble) else "Benign"
        print("🛡️ File Scan Result:", final_prediction)
    except Exception as e:
        print("❌ Error scanning file:", e)

def start_file_monitor(path, scaler, encoder, categorical_features, best_xgb, best_rf, lstm_model, meta_model, drop_columns):
    handler = FileMonitorHandler(scaler, encoder, categorical_features, best_xgb, best_rf, lstm_model, meta_model, drop_columns)
    observer = Observer()
    observer.schedule(handler, path, recursive=True)
    observer.start()
    try:
        while True:
            pass
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# =============================================================================
# 13. Random Sample Testing
# =============================================================================
def scan_random_sample(X_test_enc, X_test, best_xgb, best_rf, lstm_model, meta_model):
    random_index = random.randint(0, X_test_enc.shape[0] - 1)
    sample = X_test_enc[random_index].reshape(1, -1)
    sample_lstm = sample.reshape(1, 1, -1)
    pred, _ = ensemble_predict(sample, sample_lstm, best_xgb, best_rf, lstm_model, meta_model, threshold=0.5)
    final_prediction = "Ransomware" if pred[0] == 1 else "Benign"
    print("📝 Random File Classified as:", final_prediction)

# =============================================================================
# 14. Main Execution Pipeline
# =============================================================================
def main():
    dataset_path = "C:\\Users\\haroo\\TETREX\\Obfuscated-MalMem2022.parquet"
    drop_columns = ["FileName", "Timestamp"]
    
    # Load and preprocess data
    df = load_dataset(dataset_path)
    X, y, scaler, categorical_features, feature_names = preprocess_data(df, drop_columns=drop_columns)
    
    # Check data distribution to detect outliers
    check_data_distribution(X, feature_names)
    
    # Split data and augment training set
    X_train, X_test, y_train, y_test = split_and_augment(X, y)
    
    # Train autoencoder (set denoising=True to test denoising autoencoder behavior)
    encoder = train_autoencoder(X_train, X_test, denoising=True, noise_std=0.1)
    X_train_enc = encoder.predict(X_train)
    X_test_enc = encoder.predict(X_test)
    
    # Train base models on encoded features
    best_xgb, best_rf = train_base_models(X_train_enc, y_train)
    
    # Train LSTM model for behavioral analysis
    lstm_model = train_lstm_model(X_train_enc, y_train, X_test_enc, y_test)
    X_test_lstm = X_test_enc.reshape(X_test_enc.shape[0], 1, X_test_enc.shape[1])
    
    # Train stacking meta-model using base model outputs
    meta_model = train_stacking_ensemble(X_train_enc, 
                                         X_train_enc.reshape(X_train_enc.shape[0], 1, X_train_enc.shape[1]),
                                         y_train, best_xgb, best_rf, lstm_model)
    
    # Evaluate ensemble on test set
    ensemble_preds, ensemble_probs = ensemble_predict(X_test_enc, X_test_lstm, best_xgb, best_rf, lstm_model, meta_model, threshold=0.5)
    print("✅ Ensemble (Stacking) Accuracy:", accuracy_score(y_test, ensemble_preds))
    print("📊 Ensemble Classification Report:\n", classification_report(y_test, ensemble_preds))
    print("Test ROC-AUC Score:", roc_auc_score(y_test, ensemble_probs))
    
    # Save models and preprocessors
    save_models(best_xgb, best_rf, meta_model, lstm_model, encoder, scaler)
    
    # Test a random sample
    scan_random_sample(X_test_enc, X_test, best_xgb, best_rf, lstm_model, meta_model)
    
    # Optionally, start real-time monitoring:
    # start_file_monitor("path_to_watch", scaler, encoder, categorical_features, best_xgb, best_rf, lstm_model, meta_model, drop_columns)

if __name__ == '__main__':
    main()
