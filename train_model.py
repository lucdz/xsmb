#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_model.py — Huấn luyện mô hình softmax 100 lớp (00..99) dự đoán 2 số cuối GĐB XSMB.
Fixes:
  - Bỏ các hàng có "Mã" khỏi all_last2 (khớp inference của app)
  - Chống trùng ngày giữa trang tổng và trang từng ngày
  - Sắp xếp theo thời gian (oldest -> newest) trước khi tạo cửa sổ 7 ngày
  - Chia train/val theo thời gian, tắt shuffle, thêm metric top-10
"""

import os
import re
import json
import hashlib
import datetime as dt
import time
import random
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -------------------- cấu hình & seed --------------------

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

OUT_MODEL   = Path("model.tflite")
OUT_VERSION = Path("version.json")
UA = {"User-Agent": "Mozilla/5.0 (XSMB-TrainingBot/3.0)"}

BASE_LIST_URL = "https://xosodaiphat.com/xsmb-xo-so-mien-bac.html"
DAY_URL_TPL   = "https://xosodaiphat.com/xsmb-{ddmmyyyy}.html"

# -------------------- helpers chung --------------------

def last2(s: str) -> str | None:
    s = re.sub(r"\D", "", s or "")
    return s[-2:] if len(s) >= 2 else None

def fetch_html(url: str, retries: int = 5, timeout: int = 20) -> str:
    last_exc = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_exc = e
            time.sleep(2 ** i)
    raise last_exc

def id_to_iso(id_str: str) -> str | None:
    # id: "kqngay_07082025" -> "2025-08-07"
    m = re.search(r'(\d{2})(\d{2})(\d{4})$', id_str or "")
    if not m: return None
    dd, mm, yyyy = m.groups()
    return f"{yyyy}-{mm}-{dd}"

def ddmmyyyy_to_iso(ddmmyyyy: str) -> str | None:
    # "07-08-2025" -> "2025-08-07"
    m = re.match(r'(\d{2})-(\d{2})-(\d{4})$', ddmmyyyy or "")
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else None

# -------------------- parsing bảng kết quả --------------------

def parse_table(tbl) -> dict | None:
    """
    Trả về:
      {"all_last2": [27 items "xy"], "gdb_last2": "xy"}
    - BỎ các hàng có "mã" khỏi all_last2 (để khớp app)
    - Ưu tiên <span>, fallback split text
    """
    all_last2, gdb_last2 = [], None

    for tr in tbl.select("tbody > tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue

        key = (tds[0].get_text(strip=True) or "").lower()

        # Ưu tiên spans; fallback tách text
        spans = [s.get_text(strip=True) for s in tds[1].select("span")]
        vals = spans if spans else re.split(r"[^0-9]+", tds[1].get_text(" ", strip=True))
        vals = [v for v in vals if v]

        # Bỏ "mã" khỏi all_last2 (khớp getAllLast2Digits() của app)
        if "mã" not in key:
            for v in vals:
                l2 = last2(v)
                if l2:
                    all_last2.append(l2)

        # Xác định G.ĐB (cũng bỏ key có "mã")
        if "mã" not in key and (
            "g.đb" in key or "g đb" in key or "giải đặc biệt" in key
            or "gdb" in key or re.search(r"g\s*\.\s*đb", key)
        ):
            if vals:
                g = last2(vals[0])
                if g:
                    gdb_last2 = g

    if not gdb_last2:
        return None
    if len(all_last2) < 20:
        # bảng chưa đầy đủ
        return None

    # Chuẩn hoá 27 số theo app
    if len(all_last2) < 27:
        all_last2 += ["00"] * (27 - len(all_last2))
    elif len(all_last2) > 27:
        all_last2 = all_last2[:27]

    return {"all_last2": all_last2, "gdb_last2": gdb_last2}

# -------------------- scraping nhiều ngày --------------------

def scrape_days(max_days: int = 120) -> list[dict]:
    """
    Thu thập newest -> oldest, kèm 'date' ISO và chống trùng theo ngày.
    Mỗi phần tử: {"date": "YYYY-MM-DD", "all_last2": [...27...], "gdb_last2": "xy"}
    """
    out: list[dict] = []
    seen: set[str] = set()

    # 1) Trang tổng (thường ~30 ngày gần nhất)
    try:
        html = fetch_html(BASE_LIST_URL)
        soup = BeautifulSoup(html, "html.parser")
        for blk in soup.select('div[id^="kqngay_"]'):
            if len(out) >= max_days:
                break
            tbl = blk.select_one("table.table-xsmb")
            if not tbl:
                continue
            parsed = parse_table(tbl)
            d_iso = id_to_iso(blk.get("id", ""))
            if parsed and d_iso and d_iso not in seen:
                parsed["date"] = d_iso
                out.append(parsed)
                seen.add(d_iso)
    except Exception as e:
        print(f"[scrape] base page failed: {e}")

    # 2) Fallback từng ngày lùi dần (yesterday -> ...)
    today = dt.date.today()
    for i in range(1, 365):  # đủ rộng; vẫn dừng bởi max_days
        if len(out) >= max_days:
            break
        ddmmyyyy = (today - dt.timedelta(days=i)).strftime("%d-%m-%Y")
        d_iso = ddmmyyyy_to_iso(ddmmyyyy)
        if d_iso in seen:
            continue
        url = DAY_URL_TPL.format(ddmmyyyy=ddmmyyyy)
        try:
            html = fetch_html(url)
            soup = BeautifulSoup(html, "html.parser")
            tbl = soup.select_one("table.table-xsmb")
            if not tbl:
                print(f"[scrape] skip {url}: no table")
                continue
            parsed = parse_table(tbl)
            if parsed:
                parsed["date"] = d_iso
                out.append(parsed)
                seen.add(d_iso)
            else:
                print(f"[scrape] skip {url}: parsed None/short")
        except Exception as e:
            print(f"[scrape] skip {url}: {e}")

    print(f"[scrape] total unique days: {len(out)}")
    return out  # newest -> oldest (theo nguồn), sẽ sort lại theo thời gian

# -------------------- dataset --------------------

def day_to_vec54(all_last2: list[str]) -> np.ndarray:
    # 27 cặp -> 54 số; scale /9.0 như app
    arr = []
    for s in all_last2:
        arr.extend([int(s[0]), int(s[1])])
    return np.array(arr, np.float32) / 9.0  # (54,)

def make_supervised(days: list[dict], win: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """
    Input: days oldest -> newest
    Tạo mẫu: X[t] = concat(7 ngày trước t hiện tại), y[t] = gdb_last2 của ngày thứ (t)
    Trả về:
      X: (N, 378), y: (N,) nhãn 0..99 (int)
    """
    X, y = [], []
    for i in range(len(days) - win):
        past = days[i:i + win]
        nxt  = days[i + win]
        X.append(np.concatenate([day_to_vec54(d["all_last2"]) for d in past], axis=0))
        y.append(int(nxt["gdb_last2"]))  # "07" -> 7 ; "36" -> 36
    return np.stack(X, 0), np.array(y, np.int32)

# -------------------- model --------------------

def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.15),
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-5)),
        layers.Dense(64, activation='relu'),
        layers.Dense(100, activation='softmax')  # lớp 00..99
    ])
    return model

def export_tflite(model: keras.Model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    OUT_MODEL.write_bytes(converter.convert())

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()

# -------------------- main --------------------

def main():
    raw = scrape_days(120)  # newest -> oldest
    if len(raw) < 20:
        print("Not enough data scraped (<20). Abort.")
        raise SystemExit(1)

    # oldest -> newest theo date ISO
    days = sorted(raw, key=lambda d: d["date"])

    X, y = make_supervised(days, win=7)
    if len(X) < 30:
        print("Not enough supervised samples (<30). Abort.")
        raise SystemExit(1)

    print(f"[dataset] X:{X.shape} y:{y.shape}")

    model = build_model(X.shape[1])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top10')
        ]
    )

    # Chia theo thời gian, không shuffle
    split = int(len(X) * 0.85)
    cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    history = model.fit(
        X[:split], y[:split],
        validation_data=(X[split:], y[split:]),
        epochs=120, batch_size=64,
        shuffle=False, verbose=0,
        callbacks=[cb]
    )

    val_loss, val_acc, val_top10 = model.evaluate(X[split:], y[split:], verbose=0)
    print(f"[val] loss={val_loss:.4f} acc={val_acc:.4f} top10={val_top10:.4f}")

    export_tflite(model)

    ver = dt.datetime.utcnow().strftime("%Y%m%d-%H%M")
    commit = os.environ.get("GITHUB_SHA", "main")  # dùng SHA của workflow hiện tại nếu có

    OUT_VERSION.write_text(json.dumps({
        "version": ver,
        "url": f"https://cdn.jsdelivr.net/gh/lucdz/xsmb@{commit}/model.tflite",  # dùng @commit để tránh cache đổi file
        "sha256": sha256(OUT_MODEL)
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("OK", ver)

if __name__ == "__main__":
    main()
