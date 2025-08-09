# train_model.py
import re, json, hashlib, datetime, sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_URL = "https://xosodaiphat.com/xsmb-xo-so-mien-bac.html"
OUT_MODEL = Path("model.tflite")
OUT_VERSION = Path("version.json")

def fetch_last_days(n=8):
    """Trả về list [(date, gdb5digits)] mới -> cũ"""
    html = requests.get(BASE_URL, timeout=20, headers={"User-Agent":"Mozilla/5.0"}).text
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for blk in soup.select('div[id^="kqngay_"]'):
        if len(out) >= n: break
        raw = blk.get("id","").replace("kqngay_","")  # ddMMyyyy
        date = f"{raw[4:8]}-{raw[2:4]}-{raw[0:2]}"    # yyyy-MM-dd

        tbl = blk.select_one("table.table-xsmb")
        if not tbl: continue
        # tìm hàng G.ĐB (bỏ "Mã ĐB")
        gdb = None
        for tr in tbl.select("tbody > tr"):
            tds = tr.find_all("td")
            if len(tds) < 2: continue
            key = tds[0].get_text(strip=True).lower()
            if "mã" in key:  # bỏ "Mã ĐB"
                continue
            if "g.đb" in key or "g đb" in key or "giải đặc biệt" in key or "gdb" in key:
                vals = [s.get_text(strip=True) for s in tds[1].select("span")]
                if vals:
                    digits = re.sub(r"\D","", vals[0])[-5:]
                    if len(digits)==5:
                        gdb = digits
                break
        if gdb:
            out.append((date, gdb))
    return out  # mới -> cũ

def build_dataset(pairs):
    # cặp ngày liên tiếp: prev -> next
    X, y = [], []
    for i in range(len(pairs)-1):
        prev5 = [int(c) for c in pairs[i][1]]
        next5 = [int(c) for c in pairs[i+1][1]]
        X.append(prev5); y.append(next5)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def train_and_export(X, y):
    model = keras.Sequential([
        layers.Input(shape=(5,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(5, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=200, verbose=0)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    OUT_MODEL.write_bytes(tflite_model)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()

def main():
    pairs = fetch_last_days(9)  # lấy dư 1 để có 8 cặp train
    if len(pairs) < 2:
        print("Not enough data")
        sys.exit(1)

    X, y = build_dataset(pairs)
    train_and_export(X, y)

    ver = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M")  # UTC
    OUT_VERSION.write_text(json.dumps({
        "version": ver,
        "url": "https://cdn.jsdelivr.net/gh/lucdz/xsmb@main/model.tflite",
        "sha256": sha256(OUT_MODEL)
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OK", ver)

if __name__ == "__main__":
    main()
