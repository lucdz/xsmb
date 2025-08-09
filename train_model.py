# train_model.py — predict last-2 digits of the special prize (GĐB)
# using 27 last-2 digits from ALL prizes (XSMB) for the past 7 days.

import re, json, hashlib, datetime, sys, time, random
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from bs4 import BeautifulSoup
# ----------- Reproducibility -----------
random.seed(42); np.random.seed(42); tf.random.set_seed(42)

# ----------- Config -----------
BASE_URL_MAIN = "https://xosodaiphat.com/xsmb-xo-so-mien-bac.html"
BASE_URL_30   = "https://xosodaiphat.com/xsmb-30-ngay.html"
DAY_URL_FMT   = "https://xosodaiphat.com/xsmb-{dd}-{mm}-{yyyy}.html"

OUT_MODEL   = Path("model.tflite")
OUT_VERSION = Path("version.json")

UA = {"User-Agent": "Mozilla/5.0 (XSMB-TrainingBot/2.0)"}
def make_soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        # Fallback nếu lxml không có
        return BeautifulSoup(html, "html.parser")
# ========== Utils ==========

def fetch_html(url, retries=5, timeout=20):
    """GET with retry/backoff."""
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last = e
            sleep = min(2 ** i, 10)
            time.sleep(sleep)
    raise last

def last2(s: str):
    digits = re.sub(r"\D", "", s or "")
    return digits[-2:] if len(digits) >= 2 else None

def parse_one_table(tbl) -> dict | None:
    """
    Return dict:
      - 'all_last2': list[str] length 27 (last-2 of ALL prizes in table order)
      - 'gdb_last2': str (last-2 of special prize)
    If data is too sparse -> None.
    """
    all_last2 = []
    gdb_last2 = None

    for tr in tbl.select("tbody > tr"):
        tds = tr.find_all("td")
        if len(tds) < 2: continue
        key = (tds[0].get_text(strip=True) or "").lower()

        # collect all numbers in the result column
        vals = [sp.get_text(strip=True) for sp in tds[1].select("span")]
        # some sites put text directly without span
        if not vals:
            txt = tds[1].get_text(" ", strip=True)
            vals = [x for x in re.split(r"\s+", txt) if x]

        for v in vals:
            l2 = last2(v)
            if l2 is not None:
                all_last2.append(l2)

        # detect special prize row for label
        if "mã" not in key and (
            "g.đb" in key or "g đb" in key or "giải đặc biệt" in key
            or "gdb" in key or re.search(r"g\s*\.\s*đb", key)
        ):
            if vals:
                g = last2(vals[0])
                if g: gdb_last2 = g

    if gdb_last2 is None:
        return None

    # Normalize to exactly 27 numbers: pad "00" if missing, cut if too many.
    if len(all_last2) < 20:
        return None  # too sparse -> drop this day
    if len(all_last2) < 27:
        all_last2 = all_last2 + (["00"] * (27 - len(all_last2)))
    elif len(all_last2) > 27:
        all_last2 = all_last2[:27]

    return {"all_last2": all_last2, "gdb_last2": gdb_last2}

def parse_from_soup(soup: BeautifulSoup):
    """Extract days (newest->oldest) from a soup document."""
    out = []
    blocks = soup.select('div[id^="kqngay_"]')
    if not blocks:
        tbl = soup.select_one("table.table-xsmb")
        if tbl:
            parsed = parse_one_table(tbl)
            if parsed: out.append(parsed)
        return out

    for blk in blocks:
        tbl = blk.select_one("table.table-xsmb")
        if not tbl: continue
        parsed = parse_one_table(tbl)
        if parsed: out.append(parsed)
    return out

def scrape_days(target=90):
    """
    Collect as many days as possible (newest->oldest).
    1) Try '30 ngày' page.
    2) If still short, go day-by-day with xsmb-dd-mm-yyyy.html.
    """
    out = []

    # 1) 30-day page
    try:
        html = fetch_html(BASE_URL_30)
        soup = make_soup(html)
        out.extend(parse_from_soup(soup))
    except Exception as e:
        print("[scrape] 30-day page failed:", e)

    print(f"[scrape] from 30-day page: {len(out)} days")

    # 2) If still short, backfill per-day pages
    if len(out) < target:
        from datetime import datetime, timedelta
        # start from tomorrow (to ensure we step into yesterday first)
        start = datetime.utcnow() + timedelta(days=1)
        tried = 0
        while len(out) < target and tried < 240:  # try up to ~8 months back
            d = start - timedelta(days=tried+1)
            dd, mm, yyyy = d.strftime("%d"), d.strftime("%m"), d.strftime("%Y")
            url = DAY_URL_FMT.format(dd=dd, mm=mm, yyyy=yyyy)
            try:
                html = fetch_html(url)
                soup = make_soup(html)
                got = parse_from_soup(soup)
                if got:
                    out.extend(got)
                    print(f"[scrape] +1 day via {url} (total {len(out)})")
            except Exception as e:
                print(f"[scrape] skip {url}: {e}")
            finally:
                tried += 1

    print(f"[scrape] total collected: {len(out)} days")
    return out[:target]  # newest->oldest

# ---------- Dataset ----------

def digits_to_vector(two_digits: str) -> np.ndarray:
    # "xy" -> [x, y] scaled /9
    return np.array([int(two_digits[0]), int(two_digits[1])], dtype=np.float32) / 9.0

def day_to_vector_54(all_last2: list[str]) -> np.ndarray:
    # 27 numbers × 2 digits -> 54 digits, scaled /9
    arr = []
    for s in all_last2:
        arr.append(int(s[0])); arr.append(int(s[1]))
    return np.array(arr, dtype=np.float32) / 9.0  # (54,)

def make_supervised(days, win=7):
    """
    days: oldest->newest
    X: [samples, win*54] ; y: [samples, 2]
    """
    X, y = [], []
    for i in range(len(days) - win):
        past = days[i:i+win]
        nxt  = days[i+win]
        x = np.concatenate([day_to_vector_54(d["all_last2"]) for d in past], axis=0)
        X.append(x)
        y.append(digits_to_vector(nxt["gdb_last2"]))
    return np.stack(X, 0), np.stack(y, 0)

# ---------- Model ----------

def build_model(input_dim: int):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='sigmoid')  # 0..1 -> *9 -> round to digit
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

def export_tflite(model):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = conv.convert()
    OUT_MODEL.write_bytes(tflite_model)

def sha256(p: Path) -> str:
    h = hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

# ---------- Main ----------

def main():
    # try to get ~60 days; usually enough even with sparse days
    raw = scrape_days(target=60)  # newest->oldest
    if len(raw) < 10:
        print("Not enough data scraped (<10). Abort.")
        sys.exit(1)

    # oldest->newest for sliding window
    days = list(reversed(raw))

    X, y = make_supervised(days, win=7)  # input 378, output 2
    if len(X) < 10:
        print("Not enough supervised samples (<10). Abort.")
        sys.exit(1)

    print(f"[dataset] X:{X.shape} y:{y.shape}")

    model = build_model(X.shape[1])  # 378
    cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(
        X, y, validation_split=0.15,
        epochs=120, batch_size=64, verbose=0, callbacks=[cb]
    )

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

