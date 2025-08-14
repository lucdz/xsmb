#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_model.py — CNN (7×27×2) + calibration, output 100 lớp.
- Scrape sâu (đến ~5 năm), bỏ 'Mã', chuẩn hóa đủ 27 số
- Chia theo thời gian, tắt shuffle
- Kiến trúc: Reshape→Conv2D/SeparableConv2D + GAP + Dense
- Label smoothing + EarlyStopping + ReduceLROnPlateau
- Temperature scaling (calibration) trước khi xuất TFLite
- Input TFLite vẫn là [1, 378] float32 (app KHÔNG cần đổi)
"""

import os, re, json, hashlib, time, random, datetime as dt
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

random.seed(42); np.random.seed(42); tf.random.set_seed(42)

OUT_MODEL   = Path("model.tflite")
OUT_VERSION = Path("version.json")
UA = {"User-Agent": "Mozilla/5.0 (XSMB-TrainingBot/5.0)"}

BASE_LIST_URL = "https://xosodaiphat.com/xsmb-xo-so-mien-bac.html"
DAY_URL_TPL   = "https://xosodaiphat.com/xsmb-{ddmmyyyy}.html"

# ---------- helpers ----------
def last2(s): s=re.sub(r"\D","",s or ""); return s[-2:] if len(s)>=2 else None
def fetch_html(url, retries=5, timeout=20):
    last=None
    for i in range(retries):
        try:
            r=requests.get(url, headers=UA, timeout=timeout); r.raise_for_status(); return r.text
        except Exception as e:
            last=e; time.sleep(2**i)
    raise last
def id_to_iso(id_str):
    m=re.search(r'(\d{2})(\d{2})(\d{4})$', id_str or ""); 
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else None
def ddmmyyyy_to_iso(ddmmyyyy):
    m=re.match(r'(\d{2})-(\d{2})-(\d{4})$', ddmmyyyy or "")
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else None

def parse_table(tbl):
    all_last2, gdb_last2 = [], None
    for tr in tbl.select("tbody > tr"):
        tds = tr.find_all("td")
        if len(tds) < 2: continue
        key = (tds[0].get_text(strip=True) or "").lower()
        spans = [s.get_text(strip=True) for s in tds[1].select("span")]
        vals = spans if spans else re.split(r"[^0-9]+", tds[1].get_text(" ", strip=True))
        vals = [v for v in vals if v]

        if "mã" not in key:
            for v in vals:
                l2 = last2(v)
                if l2: all_last2.append(l2)

        if "mã" not in key and (
            "g.đb" in key or "g đb" in key or "giải đặc biệt" in key
            or "gdb" in key or re.search(r"g\s*\.\s*đb", key)
        ):
            if vals:
                g = last2(vals[0])
                if g: gdb_last2 = g

    if not gdb_last2 or len(all_last2) < 20: return None
    if len(all_last2) < 27: all_last2 += ["00"]*(27-len(all_last2))
    elif len(all_last2) > 27: all_last2 = all_last2[:27]
    return {"all_last2": all_last2, "gdb_last2": gdb_last2}

def scrape_days(max_days=1500):  # ~4+ năm
    out, seen = [], set()
    # trang tổng
    try:
        soup = BeautifulSoup(fetch_html(BASE_LIST_URL), "html.parser")
        for blk in soup.select('div[id^="kqngay_"]'):
            if len(out) >= max_days: break
            tbl = blk.select_one("table.table-xsmb")
            if not tbl: continue
            parsed = parse_table(tbl)
            d_iso = id_to_iso(blk.get("id", ""))
            if parsed and d_iso and d_iso not in seen:
                parsed["date"] = d_iso; out.append(parsed); seen.add(d_iso)
    except Exception as e:
        print(f"[scrape] base page failed: {e}")

    # từng ngày lùi dần
    today = dt.date.today()
    for i in range(1, 4000):
        if len(out) >= max_days: break
        ddmmyyyy = (today - dt.timedelta(days=i)).strftime("%d-%m-%Y")
        d_iso = ddmmyyyy_to_iso(ddmmyyyy)
        if d_iso in seen: continue
        url = DAY_URL_TPL.format(ddmmyyyy=ddmmyyyy)
        try:
            soup = BeautifulSoup(fetch_html(url), "html.parser")
            tbl = soup.select_one("table.table-xsmb")
            if not tbl: continue
            parsed = parse_table(tbl)
            if parsed:
                parsed["date"] = d_iso; out.append(parsed); seen.add(d_iso)
        except Exception as e:
            print(f"[scrape] skip {url}: {e}")
    print(f"[scrape] total unique days: {len(out)}")
    return out

def day_to_vec54(all_last2):
    a=[]; 
    for s in all_last2: a.extend([int(s[0]), int(s[1])])
    return (np.array(a, np.float32) / 9.0)

def make_supervised(days, win=7):
    X, y = [], []
    for i in range(len(days)-win):
        past = days[i:i+win]; nxt=days[i+win]
        X.append(np.concatenate([day_to_vec54(d["all_last2"]) for d in past], axis=0))  # (378,)
        y.append(int(nxt["gdb_last2"]))
    return np.stack(X,0), np.array(y, np.int32)

# ---------- model ----------
def conv_block(x, f, k=(3,3), dw=False, name=None):
    if dw:
        x = layers.SeparableConv2D(f, k, padding="same", activation=None, name=name)(x)
    else:
        x = layers.Conv2D(f, k, padding="same", activation=None, name=name)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def build_model(input_dim):
    reg = keras.regularizers.l2(1e-5)
    inp = keras.Input(shape=(input_dim,), name="flat378")
    x = layers.Reshape((7,27,2), name="reshape_7x27x2")(inp)

    x = conv_block(x, 32)                           # local pattern
    x = conv_block(x, 32, dw=True)
    x = layers.MaxPooling2D(pool_size=(1,3))(x)     # nén theo trục 27

    x = conv_block(x, 48)
    x = conv_block(x, 64, dw=True)
    x = layers.MaxPooling2D(pool_size=(2,1))(x)     # nén theo trục 7 ngày

    x = conv_block(x, 64)
    x = conv_block(x, 96, dw=True)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    logits = layers.Dense(100, activation=None, name="logits")(x)  # <-- logits 100 lớp

    return keras.Model(inp, logits, name="xsmb_cnn")

def export_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    OUT_MODEL.write_bytes(converter.convert())

def sha256(p): 
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def softmax(z):
    z=z - np.max(z, axis=1, keepdims=True)
    e=np.exp(z); return e/np.sum(e, axis=1, keepdims=True)

def find_temperature(logits, y, grid=np.linspace(0.5, 3.0, 41)):
    bestT, bestLoss = 1.0, 1e9
    for T in grid:
        p = softmax(logits / T)
        eps = 1e-9
        nll = -np.mean(np.log(p[np.arange(len(y)), y] + eps))
        if nll < bestLoss:
            bestLoss, bestT = nll, T
    return float(bestT)

def sparse_ce_with_label_smoothing(num_classes=100, epsilon=0.05):
    """Loss cho nhãn sparse nhưng có label smoothing."""
    if epsilon <= 0:
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_one  = tf.one_hot(y_true, num_classes)
        y_smooth = y_one * (1.0 - epsilon) + (epsilon / num_classes)
        return cce(y_smooth, y_pred)
    return loss_fn

# ---------- main ----------
def main():
    raw = scrape_days(1500)
    if len(raw) < 80: raise SystemExit("Too few days scraped.")
    days = sorted(raw, key=lambda d: d["date"])  # oldest -> newest

    X, y = make_supervised(days, win=7)
    if len(X) < 80: raise SystemExit("Too few samples.")

    print(f"[dataset] X:{X.shape} y:{y.shape}")
    split = int(len(X)*0.85)

    model = build_model(X.shape[1])

    # NEW
    loss = sparse_ce_with_label_smoothing(num_classes=100, epsilon=0.05)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=loss,
        metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name="top10")]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=loss,
        metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name="top10")]
    )
    cbs = [
        keras.callbacks.ReduceLROnPlateau(patience=6, factor=0.5, min_lr=5e-5, verbose=1),
        keras.callbacks.EarlyStopping(patience=14, restore_best_weights=True, verbose=1)
    ]
    model.fit(
        X[:split], y[:split],
        validation_data=(X[split:], y[split:]),
        epochs=200, batch_size=256, shuffle=False, callbacks=cbs, verbose=2
    )

    # calibration
    logits_val = model.predict(X[split:], batch_size=512, verbose=0)
    T = find_temperature(logits_val, y[split:])
    print(f"[calibration] best T = {T:.3f}")

    inp = keras.Input(shape=(X.shape[1],))
    z = model(inp)                         # logits
    z = layers.Lambda(lambda t: t / T, name="temp")(z)
    prob = layers.Activation("softmax", name="softmax")(z)
    export_model = keras.Model(inp, prob)

    p_val = export_model.predict(X[split:], batch_size=512, verbose=0)
    top10 = tf.keras.metrics.sparse_top_k_categorical_accuracy(y[split:], p_val, k=10)
    print(f"[val] top10={np.mean(top10):.4f}")

    export_tflite(export_model)

    ver = dt.datetime.utcnow().strftime("%Y%m%d-%H%M")
    commit = os.environ.get("GITHUB_SHA", "main")
    OUT_VERSION.write_text(json.dumps({
        "version": ver,
        "url": f"https://cdn.jsdelivr.net/gh/lucdz/xsmb@{commit}/model.tflite",
        "sha256": sha256(OUT_MODEL)
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OK", ver)

if __name__ == "__main__":
    main()

