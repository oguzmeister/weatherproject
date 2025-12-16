import os
import threading
import time
import webbrowser
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Arayüz hatası almamak için backend ayarı
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from PIL import Image

# ---------------- CONFIG ----------------
app = Flask(__name__)
app.secret_key = "pink-weather-secret"

# Yollar (Proje dizinine göre ayarlı)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Multi-class Weather Dataset")
MODEL_PATH = os.path.join(BASE_DIR, "weather_model.h5")

STATIC_DIR = os.path.join(BASE_DIR, "static")
PLOTS_DIR = os.path.join(STATIC_DIR, "plots")
SAMPLES_DIR = os.path.join(STATIC_DIR, "test_samples")

# Klasörleri oluştur
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Parametreler
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

# Global Durum Değişkeni
training_status = {
    "is_training": False,
    "progress": 0,
    "epoch": 0,
    "total_epochs": EPOCHS,
    "message": "Hazır"
}

# ---------------- YARDIMCI FONKSİYONLAR ----------------

def get_dataset():
    """Dataset'i yükler ve train/val olarak ayırır."""
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Dataset bulunamadı: {DATA_DIR}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int"
    )
    
    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, class_names

def build_model(num_classes):
    """CNN Model Mimarisi"""
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = tf.keras.layers.Rescaling(1./255)(inputs)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

class TrainingCallback(tf.keras.callbacks.Callback):
    """Eğitim ilerlemesini arayüze aktarır."""
    def on_epoch_end(self, epoch, logs=None):
        training_status["epoch"] = epoch + 1
        training_status["progress"] = int((epoch + 1) / EPOCHS * 100)
        training_status["message"] = f"Epoch {epoch + 1}/{EPOCHS} tamamlandı. Loss: {logs['loss']:.4f}"

# ---------------- GRAFİK ÇİZİM ----------------

def plot_training_curves(history):
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    plt.figure(figsize=(10, 5))
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist["loss"], label="Eğitim Kaybı", color="#d63384")
    plt.plot(epochs, hist["val_loss"], label="Doğrulama Kaybı", color="#6f42c1")
    plt.title("Kayıp (Loss)")
    plt.legend()
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist["accuracy"], label="Eğitim Doğruluğu", color="#d63384")
    plt.plot(epochs, hist["val_accuracy"], label="Doğrulama Doğruluğu", color="#6f42c1")
    plt.title("Doğruluk (Accuracy)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_curves.png"))
    plt.close()

def plot_confusion_matrix_and_roc(model, val_ds, class_names):
    y_true, y_prob = [], []
    for img, lbl in val_ds:
        y_true.append(lbl.numpy())
        y_prob.append(model.predict(img, verbose=0))
    
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = np.argmax(y_prob, axis=1)

    # CM
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="RdPu", xticklabels=class_names, yticklabels=class_names)
    plt.title("Karmaşıklık Matrisi")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plt.close()

    # ROC
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    fpr, tpr, roc_auc = {}, {}, {}
    n_classes = len(class_names)
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(8, 7))
    plt.plot(fpr["micro"], tpr["micro"], label=f"Micro-avg (AUC={roc_auc['micro']:.2f})", color="#d63384", linestyle=':', linewidth=4)
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC={roc_auc[i]:.2f})")
    
    plt.plot([0,1],[0,1],'k--')
    plt.title("ROC Eğrisi")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"))
    plt.close()

def save_test_samples(model, val_ds, class_names, num_samples=8):
    for f in os.listdir(SAMPLES_DIR): os.remove(os.path.join(SAMPLES_DIR, f))
    
    images, labels = next(iter(val_ds.unbatch().batch(num_samples)))
    preds = np.argmax(model.predict(images, verbose=0), axis=1)
    
    samples = []
    for i in range(len(images)):
        img_arr = images[i].numpy().astype("uint8")
        Image.fromarray(img_arr).save(os.path.join(SAMPLES_DIR, f"sample_{i}.png"))
        samples.append({
            "path": f"static/test_samples/sample_{i}.png",
            "true": class_names[labels[i]],
            "pred": class_names[preds[i]],
            "correct": labels[i] == preds[i]
        })
    return samples

# ---------------- ARKA PLAN THREAD ----------------

def train_background():
    global training_status
    try:
        training_status["is_training"] = True
        training_status["progress"] = 0
        training_status["message"] = "Veri seti hazırlanıyor..."
        train_ds, val_ds, class_names = get_dataset()
        
        training_status["message"] = "Model derleniyor..."
        model = build_model(len(class_names))
        
        training_status["message"] = "Eğitim başladı..."
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[TrainingCallback()])
        
        training_status["message"] = "Grafikler çiziliyor..."
        plot_training_curves(history)
        model.save(MODEL_PATH)
        plot_confusion_matrix_and_roc(model, val_ds, class_names)
        
        training_status["progress"] = 100
        training_status["message"] = "Tamamlandı!"
    except Exception as e:
        training_status["message"] = f"Hata: {str(e)}"
    finally:
        training_status["is_training"] = False

# ---------------- ROUTE'LAR ----------------

@app.route("/")
def index():
    has_plots = os.path.exists(os.path.join(PLOTS_DIR, "training_curves.png"))
    return render_template("index.html", has_plots=has_plots)

@app.route("/train", methods=["POST"])
def train():
    if training_status["is_training"]: return jsonify({"status": "error", "message": "Zaten çalışıyor"})
    threading.Thread(target=train_background).start()
    return jsonify({"status": "started"})

@app.route("/status")
def status(): return jsonify(training_status)

@app.route("/test", methods=["POST"])
def test():
    if not os.path.exists(MODEL_PATH):
        flash("Önce eğitim yapmalısınız!", "error")
        return redirect(url_for("index"))
    
    model = tf.keras.models.load_model(MODEL_PATH)
    _, val_ds, class_names = get_dataset()
    plot_confusion_matrix_and_roc(model, val_ds, class_names)
    samples = save_test_samples(model, val_ds, class_names)
    return render_template("index.html", has_plots=True, samples=samples)

if __name__ == "__main__":
    def open_browser():
        time.sleep(1)
        webbrowser.open("http://127.0.0.1:5000")
    threading.Thread(target=open_browser).start()
    app.run(debug=True)