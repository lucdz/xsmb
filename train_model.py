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

CẬP NHẬT (fix job fail liên tục + chạy ~1 tiếng rồi mới báo lỗi trên GitHub Actions):
1) Cache lịch sử vào data/xsmb_history.json — mỗi lần chạy chỉ scrape thêm vài
   ngày MỚI thay vì dò lại toàn bộ lịch sử từ đầu mỗi ngày.
   => Workflow .yml cần thêm bước commit lại file cache này sau khi chạy, xem
      ghi chú cuối file, nếu không cache sẽ mất giữa các lần chạy.
2) Log chi tiết HTTP status / có tìm thấy table hay không -> biết chính xác
   chỗ nào fail nếu vẫn lỗi (thay vì phải đoán).
3) Fail-fast: nếu 10 lần scrape liên tiếp đều fail thì dừng ngay (vài giây)
   thay vì cố chạy hết rồi mới báo lỗi sau ~1 tiếng.
4) User-Agent giống trình duyệt thật thay vì tự khai "...TrainingBot..."
   (chuỗi cũ rất dễ bị WAF/Cloudflare nhận diện là bot).
5) Có delay nhỏ giữa các request.
6) FIX GỐC RỄ (phát hiện từ log run #112): table.table-xsmb vẫn tồn tại và
   fetch OK (HTTP 200), nhưng parse_table cũ dùng re.split để tách số trong
   mỗi giải — trong khi site hiện nối các số trong CÙNG 1 giải dính liền
   nhau không dấu cách (vd G.7 "99350386" = 4 số 2 chữ số, không phải 1 số
   8 chữ số). Sửa lại: cắt theo ĐỘ DÀI CỐ ĐỊNH của từng giải (G.ĐB/G.1/G.2/
   G.3=5, G.4/G.5=4, G.6=3, G.7=2). Có debug in chi tiết 2 trang đầu mỗi lần
   chạy để phát hiện sớm nếu site đổi định dạng lần nữa.
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
CACHE_FILE  = Path("data/xsmb_history.json")   # MỚI: lịch sử được giữ lại giữa các lần chạy

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}

BASE_LIST_URL = "https://xosodaiphat.com/xsmb-xo-so-mien-bac.html"
DAY_URL_TPL   = "https://xosodaiphat.com/xsmb-{ddmmyyyy}.html"

REQUEST_DELAY     = 0.35  # giây, nghỉ giữa các request để đỡ bị nghi là bot
STOP_AFTER_KNOWN  = 5     # gặp liên tiếp N ngày đã có trong cache -> coi như bắt kịp lịch sử
FAIL_FAST_AFTER   = 10    # N lần scrape liên tiếp đều fail -> dừng sớm luôn

# ---------- helpers ----------
def last2(s):
    s = re.sub(r"\D", "", s or "")
    return s[-2:] if len(s) >= 2 else None

def fetch_html(url, retries=5, timeout=20):
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            print(f"[fetch] {url} -> HTTP {r.status_code} ({len(r.content)} bytes)")
            r.raise_for_status()
            return r.text
        except Exception as e:
            last = e
            print(f"[fetch] lỗi lần {i+1}/{retries} tại {url}: {e}")
            time.sleep(2 ** i)
    raise last

# ---------- cache lịch sử (MỚI) ----------
def load_cache():
    if not CACHE_FILE.exists():
        print(f"[cache] không thấy {CACHE_FILE}, bắt đầu từ rỗng")
        return []
    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        print(f"[cache] đã load {len(data)} ngày từ {CACHE_FILE}")
        return data
    except Exception as e:
        print(f"[cache] lỗi đọc cache ({e}), bỏ qua và bắt đầu từ rỗng")
        return []

def save_cache(days):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(days, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[cache] đã lưu {len(days)} ngày vào {CACHE_FILE}")

# XSMB: mỗi giải có ĐỘ DÀI CỐ ĐỊNH cho mỗi số (site nối các số trong cùng 1 giải
# dính liền nhau, KHÔNG có dấu cách/span riêng — vd G.7 có 4 số 2 chữ số hiện thành
# 1 chuỗi 8 chữ số "99350386" chứ không phải 4 span tách rời). Do đó cách đúng là
# cắt chuỗi số theo đúng độ dài mỗi giải, không phải re.split theo ký tự phân cách.
_TIER_PATTERNS = [
    ("gdb", re.compile(r"g\s*\.?\s*đ\s*b|giải\s*đặc\s*biệt"), 5),
    ("g1",  re.compile(r"g\s*\.?\s*1\b"), 5),
    ("g2",  re.compile(r"g\s*\.?\s*2\b"), 5),
    ("g3",  re.compile(r"g\s*\.?\s*3\b"), 5),
    ("g4",  re.compile(r"g\s*\.?\s*4\b"), 4),
    ("g5",  re.compile(r"g\s*\.?\s*5\b"), 4),
    ("g6",  re.compile(r"g\s*\.?\s*6\b"), 3),
    ("g7",  re.compile(r"g\s*\.?\s*7\b"), 2),
]

def match_tier(key):
    for name, pat, width in _TIER_PATTERNS:
        if pat.search(key):
            return name, width
    return None, None

def parse_table(tbl, debug=False):
    all_last2, gdb_last2 = [], None
    # LƯU Ý: dùng "tr" chứ không phải "tbody > tr" — HTML nguồn của site không có
    # thẻ <tbody> tường minh (trình duyệt tự chèn lúc hiển thị, nhưng html.parser
    # của bs4 thì không tự thêm), nên "tbody > tr" luôn khớp 0 hàng dù bảng vẫn có.
    # Hàng lạ (vd header) vẫn an toàn vì match_tier() sẽ bỏ qua nếu không khớp giải nào.
    if debug:
        print(f"[parse] số hàng <tr> tìm được trong bảng: {len(tbl.select('tr'))}")
    for tr in tbl.select("tr"):
        tds = tr.find_all("td")
        if len(tds) < 2: continue
        key = (tds[0].get_text(strip=True) or "").lower()
        if "mã" in key:
            continue

        tier, width = match_tier(key)
        if tier is None:
            if debug:
                print(f"[parse] key={key!r} không khớp giải nào đã biết -> bỏ qua hàng")
            continue

        raw_text = tds[1].get_text(" ", strip=True)
        digits = re.sub(r"\D", "", raw_text)

        if digits and len(digits) % width == 0:
            vals = [digits[i:i+width] for i in range(0, len(digits), width)]
        else:
            # dự phòng: nếu độ dài không chia hết (format khác dự kiến), thử theo
            # span (nếu có) hoặc tách theo dấu phân cách như cách cũ
            spans = [s.get_text(strip=True) for s in tds[1].select("span")]
            vals = spans if spans else re.split(r"[^0-9]+", raw_text)
            vals = [v for v in vals if v]

        if debug:
            print(f"[parse] key={key!r} tier={tier} width={width} digits_len={len(digits)} -> vals={vals}")

        for v in vals:
            l2 = last2(v)
            if l2: all_last2.append(l2)

        if tier == "gdb" and vals:
            g = last2(vals[0])
            if g: gdb_last2 = g

    if debug:
        print(f"[parse] tổng all_last2={len(all_last2)} gdb_last2={gdb_last2}")

    # chỉ nhận mẫu sạch: đủ 27 số & có GĐB
    if not gdb_last2 or len(all_last2) != 27:
        return None
    return {"all_last2": all_last2, "gdb_last2": gdb_last2}

_DEBUG_PAGES_LEFT = [2]  # in chi tiết từng hàng cho 2 trang ĐẦU TIÊN mỗi lần chạy (theo dõi lâu dài)

def scrape_day(ddmmyyyy):
    """Scrape đúng 1 ngày. Trả None nếu không lấy được / không hợp lệ (đã log lý do)."""
    url = DAY_URL_TPL.format(ddmmyyyy=ddmmyyyy)
    try:
        soup = BeautifulSoup(fetch_html(url), "html.parser")
        tbl = soup.select_one("table.table-xsmb")
        if not tbl:
            print(f"[scrape] KHÔNG thấy table.table-xsmb tại {url} (site đổi cấu trúc hoặc bị chặn?)")
            return None
        debug = _DEBUG_PAGES_LEFT[0] > 0
        if debug:
            _DEBUG_PAGES_LEFT[0] -= 1
            print(f"[scrape] --- debug chi tiết cho {url} ---")
        parsed = parse_table(tbl, debug=debug)
        if not parsed:
            print(f"[scrape] có table nhưng parse thiếu (không đủ 27 số / thiếu G.ĐB): {url}")
        return parsed
    except Exception as e:
        print(f"[scrape] skip {url}: {e}")
        return None

def scrape_incremental(existing_days, max_days=1500, max_new=45):
    """
    Đi lùi từ hôm nay, CHỈ scrape những ngày chưa có trong cache.
    Dừng sớm khi:
      - gặp liên tiếp STOP_AFTER_KNOWN ngày đã có sẵn (bắt kịp lịch sử), hoặc
      - đã lấy đủ max_new ngày mới, hoặc
      - FAIL_FAST_AFTER lần scrape liên tiếp đều fail (site chặn / đổi cấu trúc),
        để job fail nhanh trong vài giây thay vì chạy hết rồi mới báo lỗi.
    """
    seen = {d["date"] for d in existing_days}
    out = []
    attempted = 0
    today = dt.date.today()
    consecutive_known = 0

    for i in range(max_days):
        d_date = today - dt.timedelta(days=i)
        d_iso = d_date.isoformat()

        if d_iso in seen:
            consecutive_known += 1
            if consecutive_known >= STOP_AFTER_KNOWN:
                print(f"[scrape] đã bắt kịp lịch sử quanh {d_iso}, dừng dò thêm")
                break
            continue
        consecutive_known = 0

        if len(out) >= max_new:
            print(f"[scrape] đã đạt giới hạn {max_new} ngày mới cho lần chạy này, dừng lại")
            break

        parsed = scrape_day(d_date.strftime("%d-%m-%Y"))
        attempted += 1
        if parsed:
            parsed["date"] = d_iso
            out.append(parsed)
            print(f"[scrape] OK {d_iso} (mới lấy được: {len(out)})")
        elif attempted >= FAIL_FAST_AFTER and len(out) == 0:
            print(f"[scrape] {attempted} lần scrape liên tiếp đều fail -> dừng sớm. "
                  f"Kiểm tra log [fetch]/[scrape] phía trên (HTTP status? "
                  f"table.table-xsmb có tồn tại không? site đổi cấu trúc hay chặn IP/UA?)")
            break

        time.sleep(REQUEST_DELAY)

    print(f"[scrape] tổng số ngày MỚI lấy được lần này: {len(out)}")
    return out

def day_to_vec54(all_last2):
    a = []
    for s in all_last2:
        a.extend([int(s[0]), int(s[1])])
    return np.array(a, np.float32) / 9.0

def make_supervised(days, win=7):
    X, y = [], []
    for i in range(len(days) - win):
        past = days[i:i+win]; nxt = days[i+win]
        X.append(np.concatenate([day_to_vec54(d["all_last2"]) for d in past], axis=0))  # (378,)
        y.append(int(nxt["gdb_last2"]))
    return np.stack(X, 0), np.array(y, np.int32)

# ---------- model (giữ nguyên) ----------
def conv_block(x, f, k=(3, 3), dw=False, name=None):
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
    x = layers.Reshape((7, 27, 2), name="reshape_7x27x2")(inp)

    x = conv_block(x, 32)
    x = conv_block(x, 32, dw=True)
    x = layers.MaxPooling2D(pool_size=(1, 3))(x)

    x = conv_block(x, 48)
    x = conv_block(x, 64, dw=True)
    x = layers.MaxPooling2D(pool_size=(2, 1))(x)

    x = conv_block(x, 64)
    x = conv_block(x, 96, dw=True)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    logits = layers.Dense(100, activation=None, name="logits")(x)

    return keras.Model(inp, logits, name="xsmb_cnn")

def export_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    OUT_MODEL.write_bytes(converter.convert())

def sha256(p):
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

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
    if epsilon <= 0:
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_one = tf.one_hot(y_true, num_classes)
        y_smooth = y_one * (1.0 - epsilon) + (epsilon / num_classes)
        return cce(y_smooth, y_pred)
    return loss_fn

# ---------- main ----------
def main():
    existing = load_cache()

    # Cache còn mỏng (lần đầu chạy, hoặc mọi lần trước đều fail) -> cho phép dò sâu hơn.
    # Cache đã đủ dày -> mỗi lần chỉ cần bắt thêm vài ngày mới nhất.
    max_new = 1500 if len(existing) < 80 else 45
    print(f"[main] cache hiện có {len(existing)} ngày -> max_new lần này = {max_new}")
    new_days = scrape_incremental(existing, max_days=1500, max_new=max_new)

    days_by_date = {d["date"]: d for d in existing}
    for d in new_days:
        days_by_date[d["date"]] = d
    days = sorted(days_by_date.values(), key=lambda d: d["date"])
    print(f"[dataset] tổng số ngày sau khi merge cache: {len(days)}")

    # Lưu cache NGAY, trước khi train — để nếu bước train phía dưới lỗi,
    # dữ liệu vừa scrape được vẫn không bị mất cho lần chạy sau.
    save_cache(days)

    if len(days) < 80:
        raise SystemExit(
            f"Too few days scraped/cached ({len(days)}). "
            f"Xem log [fetch]/[scrape] phía trên để biết chính xác lý do "
            f"(HTTP status, có tìm thấy table.table-xsmb không...)."
        )

    X, y = make_supervised(days, win=7)
    if len(X) < 80:
        raise SystemExit("Too few samples.")
    print(f"[dataset] X:{X.shape} y:{y.shape}")

    split = int(len(X) * 0.85)

    # ---- thống kê & weight lớp để giảm thiên lệch
    hist_train = np.bincount(y[:split], minlength=100).astype(np.float32)
    w = hist_train.mean() / np.maximum(hist_train, 1.0)   # mean / freq
    class_weight = {i: float(w[i]) for i in range(100)}
    print("[label_hist] min/max:", hist_train.min(), hist_train.max())

    model = build_model(X.shape[1])

    loss = sparse_ce_with_label_smoothing(num_classes=100, epsilon=0.01)
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
        epochs=200, batch_size=256, shuffle=False, callbacks=cbs, verbose=2,
        class_weight=class_weight
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
