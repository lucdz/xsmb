# train_model.py — predict 10 candidates of last-2 digits of GĐB
import re, json, hashlib, datetime, sys, time, random
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

random.seed(42); np.random.seed(42); tf.random.set_seed(42)

OUT_MODEL = Path("model.tflite")
OUT_VERSION = Path("version.json")
UA = {"User-Agent": "Mozilla/5.0 (XSMB-TrainingBot/3.0)"}

# -------------------- scraping --------------------

def last2(s: str) -> str:
    s = re.sub(r"\D", "", s or "")
    return s[-2:] if len(s) >= 2 else None

def parse_table(tbl) -> dict | None:
    all_last2, gdb_last2 = [], None
    # tbody/tr
    for tr in tbl.select("tbody > tr"):
        tds = tr.find_all("td")
        if len(tds) < 2: continue
        key = (tds[0].get_text(strip=True) or "").lower()
        # collect values (prefer span, else split text)
        spans = [s.get_text(strip=True) for s in tds[1].select("span")]
        vals = spans if spans else re.split(r"[^0-9]+", tds[1].get_text(" ", strip=True))
        vals = [v for v in vals if v]
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
    if not gdb_last2: return None
    if len(all_last2) < 20:  # quá thiếu, bỏ
        return None
    # chuẩn hoá 27 số
    if len(all_last2) < 27:
        all_last2 += ["00"] * (27 - len(all_last2))
    elif len(all_last2) > 27:
        all_last2 = all_last2[:27]
    return {"all_last2": all_last2, "gdb_last2": gdb_last2}

def fetch_html(url, retries=5, timeout=20):
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout); r.raise_for_status()
            return r.text
        except Exception as e:
            last = e; time.sleep(2**i)
    raise last

def scrape_days(max_days=120):
    """Collect newest -> oldest"""
    base = "https://xosodaiphat.com/xsmb-xo-so-mien-bac.html"
    out = []

    # 1) Trang tổng 30 ngày (không dùng lxml)
    try:
        html = fetch_html(base)
        soup = BeautifulSoup(html, "html.parser")
        for blk in soup.select('div[id^="kqngay_"]'):
            if len(out) >= max_days: break
            tbl = blk.select_one("table.table-xsmb")
            if not tbl: continue
            parsed = parse_table(tbl)
            if parsed: out.append(parsed)
    except Exception as e:
        print(f"[scrape] 30-day page failed: {e}")

    # 2) Fallback theo ngày DD-MM-YYYY (thêm từng ngày)
    import datetime as dt
    d = dt.date.today()
    day_links = []
    for i in range(1, 121):
        ddmmyyyy = (d - dt.timedelta(days=i)).strftime("%d-%m-%Y")
        day_links.append(f"https://xosodaiphat.com/xsmb-{ddmmyyyy}.html")

    for url in day_links:
        if len(out) >= max_days: break
        try:
            html = fetch_html(url)
            soup = BeautifulSoup(html, "html.parser")
            tbl = soup.select_one("table.table-xsmb")
            if not tbl: 
                print(f"[scrape] skip {url}: no table")
                continue
            parsed = parse_table(tbl)
            if parsed: out.append(parsed)
            else: print(f"[scrape] skip {url}: parsed None/short")
        } 
        except Exception as e:
            print(f"[scrape] skip {url}: {e}")

    print(f"[scrape] total collected: {len(out)} days")
    return out

# -------------------- dataset --------------------

def day_to_vec54(all_last2):
    arr = []
    for s in all_last2:
        arr.append(int(s[0])); arr.append(int(s[1]))
    return np.array(arr, np.float32) / 9.0  # (54,)

def make_supervised(days, win=7):
    # oldest -> newest
    X, y = [], []
    for i in range(len(days) - win):
        past = days[i:i+win]
        nxt  = days[i+win]
        X.append(np.concatenate([day_to_vec54(d["all_last2"]) for d in past], 0))  # (win*54=378)
        # target: 1 cặp -> cứ dùng, model sẽ học multi-suggestion
        y.append(np.array([int(nxt["gdb_last2"][0]), int(nxt["gdb_last2"][1])], np.float32)/9.0)
    X = np.stack(X, 0); y = np.stack(y, 0)
    return X, y

# -------------------- model --------------------

def build_model(input_dim):
    # Dense đơn giản, output 20 (10 cặp) — sigmoid để thu về [0,1], scale *9 & round
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(20, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

def export_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    OUT_MODEL.write_bytes(converter.convert())

def sha256(p: Path) -> str:
    import hashlib
    h = hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def main():
    raw = scrape_days(120)              # newest -> oldest
    if len(raw) < 20:
        print("Not enough data scraped (<20). Abort."); sys.exit(1)

    days = list(reversed(raw))          # oldest -> newest
    X, y = make_supervised(days, win=7)
    if len(X) < 30:
        print("Not enough supervised samples (<30). Abort."); sys.exit(1)

    print(f"[dataset] X:{X.shape} y:{y.shape}")  # X:(N,378) y:(N,2)

    model = build_model(X.shape[1])
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
