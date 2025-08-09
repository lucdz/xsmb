# train_model.py (compact better-baseline)
import re, json, hashlib, datetime, sys, time, random
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

random.seed(42); np.random.seed(42); tf.random.set_seed(42)

BASE_URL = "https://xosodaiphat.com/xsmb-xo-so-mien-bac.html"
OUT_MODEL = Path("model.tflite")
OUT_VERSION = Path("version.json")

UA = {"User-Agent": "Mozilla/5.0 (XSMB-TrainingBot/1.0)"}

def fetch_html(url, retries=5, timeout=20):
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            r.raise_for_status(); return r.text
        except Exception as e:
            last = e; time.sleep(2**i)
    raise last

def scrape(n=400):
    """Return list of 5-digit strings newest -> oldest."""
    html = fetch_html(BASE_URL)
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for blk in soup.select('div[id^="kqngay_"]'):
        if len(out) >= n: break
        tbl = blk.select_one("table.table-xsmb"); 
        if not tbl: continue
        gdb = None
        for tr in tbl.select("tbody > tr"):
            tds = tr.find_all("td")
            if len(tds) < 2: continue
            key = tds[0].get_text(strip=True).lower()
            if "mã" in key: continue
            if "g.đb" in key or "g đb" in key or "giải đặc biệt" in key or "gdb" in key or re.search(r"g\s*\.\s*đb", key):
                spans = [s.get_text(strip=True) for s in tds[1].select("span")]
                if spans:
                    digits = re.sub(r"\D","", spans[0])[-5:]
                    if len(digits)==5: gdb = digits
                break
        if gdb: out.append(gdb)
    return out  # newest -> oldest

def make_supervised(series, win=7):
    """series: newest->oldest  -> build sliding windows."""
    # reverse to oldest->newest cho dễ trượt
    seq = list(reversed(series))
    X, y = [], []
    for i in range(len(seq) - win):
        past = seq[i:i+win]      # win days
        nxt  = seq[i+win]        # next day
        X.append([int(c) for day in past for c in day])  # win*5 digits
        y.append([int(c) for c in nxt])
    X = np.array(X, dtype=np.float32) / 9.0
    y = np.array(y, dtype=np.float32) / 9.0
    return X, y

def build_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(5, activation='sigmoid')  # 0..1 -> *9 -> round
    ])
    model.compile(optimizer='adam', loss='mae')  # MAE ổn hơn với digit rounding
    return model

def export_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    OUT_MODEL.write_bytes(tflite_model)

def sha256(p: Path) -> str:
    import hashlib
    h = hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def main():
    series = scrape(400)
    if len(series) < 30:
        print("Not enough data scraped"); sys.exit(1)

    X, y = make_supervised(series, win=7)  # 7 ngày -> ngày sau
    n = len(X)
    idx = np.arange(n); np.random.shuffle(idx)
    split = int(n*0.85)
    tr, va = idx[:split], idx[split:]
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]

    model = build_model(X.shape[1])
    cb = keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=80, batch_size=32, verbose=0, callbacks=[cb])

    export_tflite(model)

    ver = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M")
    OUT_VERSION.write_text(json.dumps({
        "version": ver,
        "url": "https://cdn.jsdelivr.net/gh/lucdz/xsmb@main/model.tflite",
        "sha256": sha256(OUT_MODEL)
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OK", ver)

if __name__ == "__main__":
    main()
