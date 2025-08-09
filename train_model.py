# train_model.py — predict last-2 digits of special prize (GĐB)
# using 27 last-2 digits from ALL prizes (XSMB) for the past 7 days.

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

UA = {"User-Agent": "Mozilla/5.0 (XSMB-TrainingBot/2.0)"}

def fetch_html(url, retries=5, timeout=20):
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last = e; time.sleep(2**i)
    raise last

# -------- scraping helpers --------

def last2(s: str) -> str:
    digits = re.sub(r"\D", "", s)
    if len(digits) < 2: return None
    return digits[-2:]  # only last-2

def parse_one_table(tbl) -> dict:
    all_last2 = []
    gdb_last2 = None

    for tr in tbl.select("tbody > tr"):
        tds = tr.find_all("td")
        if len(tds) < 2: continue
        key = tds[0].get_text(strip=True).lower()

        vals = [sp.get_text(strip=True) for sp in tds[1].select("span")]
        for v in vals:
            l2 = last2(v)
            if l2 is not None:
                all_last2.append(l2)

        if "mã" not in key and (
            "g.đb" in key or "g đb" in key or "giải đặc biệt" in key
            or "gdb" in key or re.search(r"g\s*\.\s*đb", key)
        ):
            if vals:
                g = last2(vals[0])
                if g: gdb_last2 = g

    if gdb_last2 is None:
        return None

    # Chuẩn hoá về đúng 27 số: thiếu thì pad "00", thừa thì cắt
    if len(all_last2) < 20:
        return None  # quá thiếu → bỏ
    if len(all_last2) < 27:
        all_last2 = all_last2 + (["00"] * (27 - len(all_last2)))
    elif len(all_last2) > 27:
        all_last2 = all_last2[:27]

    return {"all_last2": all_last2, "gdb_last2": gdb_last2}

def scrape_days(n=400):
    html = fetch_html(BASE_URL)
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for blk in soup.select('div[id^="kqngay_"]'):
        if len(out) >= n: break
        tbl = blk.select_one("table.table-xsmb")
        if not tbl: continue
        parsed = parse_one_table(tbl)
        if parsed:
            out.append(parsed)
    print(f"[scrape] got {len(out)} days")
    return out

# -------- dataset building --------

def digits_to_vector(two_digits: str) -> np.ndarray:
    # two digits "xy" -> [x, y] scaled /9
    return np.array([int(two_digits[0]), int(two_digits[1])], dtype=np.float32) / 9.0

def day_to_vector_54(all_last2: list[str]) -> np.ndarray:
    """
    27 numbers, each 2 digits -> 54 digits vector (0..9), scaled /9
    Order stays as parsed.
    """
    arr = []
    for s in all_last2:
        arr.append(int(s[0])); arr.append(int(s[1]))
    return np.array(arr, dtype=np.float32) / 9.0  # shape (54,)

def make_supervised(days, win=7):
    """
    days: oldest->newest expected for sliding window.
    X: shape [samples, win*54] ; y: shape [samples, 2]
    """
    X, y = [], []
    for i in range(len(days) - win):
        past = days[i:i+win]
        nxt  = days[i+win]
        x = np.concatenate([day_to_vector_54(d["all_last2"]) for d in past], axis=0)
        X.append(x)
        y.append(digits_to_vector(nxt["gdb_last2"]))
    return np.stack(X, 0), np.stack(y, 0)

# -------- model --------

def build_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='sigmoid')  # predict 2 digits scaled 0..1
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

def export_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    OUT_MODEL.write_bytes(tflite_model)

def sha256(p: Path) -> str:
    h = hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def main():
    raw = scrape_days(400)              # newest -> oldest
    if len(raw) < 15:
        print("Not enough data scraped (<15). Abort.")
        sys.exit(1)

    days = list(reversed(raw))
    X, y = make_supervised(days, win=7)
    if len(X) < 10:
        print("Not enough supervised samples (<10). Abort.")
        sys.exit(1)

    print(f"[dataset] X:{X.shape} y:{y.shape}")

    model = build_model(X.shape[1])     # 378
    cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(X, y, validation_split=0.15,
              epochs=120, batch_size=64, verbose=0, callbacks=[cb])

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

