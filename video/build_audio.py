import asyncio, ssl, os, re, json, hashlib, subprocess, sys
import numpy as np
import edge_tts, edge_tts.communicate as ecom
import imageio_ffmpeg
ecom._SSL_CTX = ssl.create_default_context(cafile="/root/.ccr/ca-bundle.crt")
FF = imageio_ffmpeg.get_ffmpeg_exe()
SR = 44100
os.makedirs("tts", exist_ok=True)

VOICE = {
    "N": dict(voice="pt-BR-ThalitaMultilingualNeural", rate=-6, pitch="-2Hz"),
    "J": dict(voice="pt-BR-AntonioNeural", rate=-13, pitch="-3Hz"),
    "P": dict(voice="pt-BR-AntonioNeural", rate=-2, pitch="+0Hz"),
    "Q": dict(voice="pt-BR-FranciscaNeural", rate=-12, pitch="-2Hz"),
}
DEFAULT_PAUSE = {"N": 0.55, "J": 0.6, "P": 0.45, "Q": 0.8}

def parse():
    shots, items = [], []
    for raw in open("script.txt", encoding="utf-8"):
        line = raw.strip()
        if not line: continue
        if line.startswith("@"):
            parts = line[1:].split()
            params = dict(p.split("=", 1) for p in parts[1:])
            shots.append(dict(type=parts[0], params=params))
            continue
        spk, text = line.split(":", 1)
        text = text.strip(); opts = {}
        m = re.search(r"\{([^}]*)\}\s*$", text)
        if m:
            opts = dict(p.split("=", 1) for p in m.group(1).split())
            text = text[:m.start()].strip()
        items.append(dict(spk=spk, text=text, opts=opts, shot=len(shots) - 1))
    return shots, items

async def tts(item):
    spk = item["spk"]; v = VOICE[spk]
    rate = v["rate"]
    if "rate" in item["opts"]:
        rate += int(item["opts"]["rate"].replace("%", ""))
    rate_s = f"{rate:+d}%"
    vol = item["opts"].get("volume", "+0%")
    spoken = item["text"].replace("—", "...").replace("“", "").replace("”", "")
    key = hashlib.md5(f"{v['voice']}|{rate_s}|{v['pitch']}|{vol}|{spoken}".encode()).hexdigest()[:16]
    mp3, js = f"tts/{key}.mp3", f"tts/{key}.json"
    if not (os.path.exists(mp3) and os.path.exists(js) and os.path.getsize(mp3) > 0):
        for attempt in range(5):
            try:
                c = edge_tts.Communicate(spoken, v["voice"], rate=rate_s, pitch=v["pitch"], volume=vol,
                                         boundary="WordBoundary", proxy=os.environ.get("HTTPS_PROXY"))
                words = []
                with open(mp3, "wb") as f:
                    async for ch in c.stream():
                        if ch["type"] == "audio": f.write(ch["data"])
                        elif ch["type"] == "WordBoundary":
                            words.append(dict(t=ch["offset"] / 1e7, d=ch["duration"] / 1e7, text=ch["text"]))
                json.dump(words, open(js, "w"))
                break
            except Exception as e:
                print("retry", e, file=sys.stderr); await asyncio.sleep(2 + attempt * 2)
    return mp3, json.load(open(js))

def decode(mp3, filt=None):
    cmd = [FF, "-v", "error", "-i", mp3]
    if filt: cmd += ["-af", filt]
    cmd += ["-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, dtype=np.float32).copy()

def make_ir(seconds, decay, seed, bright=0.5):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR); t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-t / decay)
    # soften highs
    a = bright
    for i in range(1, n): pass
    ir = np.convolve(ir, np.ones(6) / 6, mode="same")
    ir[: int(0.012 * SR)] = 0
    return (ir / np.sqrt(np.sum(ir ** 2))).astype(np.float32)

def fconv(x, ir):
    n = len(x) + len(ir) - 1; N = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(ir, N), N)[:n].astype(np.float32)

def lowpass(x, fc):
    a = np.exp(-2 * np.pi * fc / SR)
    from itertools import accumulate
    # vectorised one-pole via lfilter-like recursion using cumulative trick is unstable; use FFT-domain filter
    N = 1 << (len(x) - 1).bit_length()
    X = np.fft.rfft(x, N); f = np.fft.rfftfreq(N, 1 / SR)
    H = 1 / np.sqrt(1 + (f / fc) ** 4)
    return np.fft.irfft(X * H, N)[: len(x)].astype(np.float32)

def bandpass(x, lo, hi):
    N = 1 << (len(x) - 1).bit_length()
    X = np.fft.rfft(x, N); f = np.fft.rfftfreq(N, 1 / SR)
    H = (1 / np.sqrt(1 + (lo / np.maximum(f, 1)) ** 4)) * (1 / np.sqrt(1 + (f / hi) ** 4))
    return np.fft.irfft(X * H, N)[: len(x)].astype(np.float32)

def movavg(x, k):
    k = max(1, int(k)); c = np.cumsum(np.concatenate([[0.0], x.astype(np.float64)]))
    y = (c[k:] - c[:-k]) / k
    pad = k // 2
    return np.concatenate([np.full(pad, y[0]), y, np.full(len(x) - len(y) - pad, y[-1])]).astype(np.float32)

def trim(a, words, thr=0.006):
    env = np.abs(a)
    idx = np.where(env > thr)[0]
    if len(idx) == 0: return a, words, 0
    s = max(0, idx[0] - int(0.03 * SR)); e = min(len(a), idx[-1] + int(0.12 * SR))
    off = s / SR
    return a[s:e], [dict(w, t=max(0, w["t"] - off)) for w in words], off

def align_words(text, words):
    """Map display tokens to reveal times using TTS word boundaries."""
    toks = [(m.start(), m.group()) for m in re.finditer(r"\S+", text)]
    norm = lambda s: re.sub(r"[^\wÀ-ÿ]", "", s.lower())
    ntext = text.lower(); pos = 0; marks = []
    for w in words:
        nw = w["text"].lower().strip()
        if not nw: continue
        i = ntext.find(nw, pos)
        if i < 0: continue
        marks.append((i, w["t"])); pos = i + len(nw)
    out = []
    for cs, tok in toks:
        tt = None
        for ci, t in marks:
            if ci <= cs + len(tok) - 1 and ci >= cs - 1: tt = t; break
        if tt is None:
            prev = [t for ci, t in marks if ci <= cs]
            tt = prev[-1] if prev else 0.0
        out.append(dict(text=tok, t=tt))
    for i in range(1, len(out)):
        out[i]["t"] = max(out[i]["t"], out[i - 1]["t"])
    return out

async def main():
    shots, items = parse()
    sem = asyncio.Semaphore(6)
    async def run(it):
        if it["spk"] == "S": return None
        async with sem: return await tts(it)
    res = await asyncio.gather(*[run(it) for it in items])

    ir_room_L = make_ir(2.6, 0.55, 1); ir_room_R = make_ir(2.6, 0.55, 2)
    ir_big_L = make_ir(4.0, 1.1, 3); ir_big_R = make_ir(4.0, 1.1, 4)

    t = 1.2; prev_shot = -1
    clips = []
    for it, r in zip(items, res):
        if it["shot"] != prev_shot:
            shots[it["shot"]]["first_item_t"] = t
            if prev_shot >= 0: t += 0.25
            shots[it["shot"]]["first_item_t"] = t
            prev_shot = it["shot"]
        if it["spk"] == "S":
            dur = float(it["opts"].get("pause", 2.2))
            it.update(start=t, dur=dur, words=[dict(text=it["text"], t=0)], pause=0)
            t += dur; continue
        mp3, words = r
        filt = None
        if it["spk"] == "P":
            filt = "asetrate=44100*0.91,aresample=44100,atempo=1.0989,equalizer=f=180:t=q:w=1:g=2,equalizer=f=3200:t=q:w=1:g=-2"
        elif it["spk"] == "J":
            filt = "equalizer=f=120:t=q:w=1:g=2,equalizer=f=4500:t=q:w=1.2:g=-2.5"
        elif it["spk"] == "Q":
            filt = "equalizer=f=3000:t=q:w=1:g=-3,highpass=f=160"
        a = decode(mp3, filt)
        a, words, _ = trim(a, words)
        a = a / (np.max(np.abs(a)) + 1e-9) * {"N": 0.62, "J": 0.72, "P": 0.70, "Q": 0.5}[it["spk"]]
        if it["opts"].get("volume"): a *= 1.12
        dur = len(a) / SR
        pause = float(it["opts"].get("pause", DEFAULT_PAUSE[it["spk"]]))
        it.update(start=t, dur=dur, pause=pause, words=align_words(it["text"], words))
        clips.append((t, a, it["spk"]))
        t += dur + pause
    total = t + 1.0

    for i, s in enumerate(shots):
        s["start"] = 0.0 if i == 0 else s["first_item_t"] - 0.3
    for i, s in enumerate(shots):
        s["end"] = shots[i + 1]["start"] if i + 1 < len(shots) else total
    n = int(total * SR) + SR * 6
    dry = np.zeros(n, np.float32); room = np.zeros(n, np.float32); big = np.zeros(n, np.float32)
    for st, a, spk in clips:
        i = int(st * SR)
        dry[i:i + len(a)] += a
        if spk in "JP": room[i:i + len(a)] += a
        if spk == "Q": big[i:i + len(a)] += a
    # voice bus stereo
    voiceL = dry + 0.16 * fconv(room, ir_room_L)[:n] + 0.35 * fconv(big, ir_big_L)[:n]
    voiceR = dry + 0.16 * fconv(room, ir_room_R)[:n] + 0.35 * fconv(big, ir_big_R)[:n]
    venv = movavg(np.abs(dry), 0.25 * SR)
    venv = np.clip(venv / 0.12, 0, 1)

    # ---------- events (sfx + visuals) ----------
    events = []
    rng = np.random.default_rng(7)
    for si, s in enumerate(shots):
        its = [it for it in items if it["shot"] == si]
        ty, mode = s["type"], s["params"].get("mode", "")
        if ty == "book":
            if mode == "open": events.append(dict(kind="book_open", t=s["start"] + 0.5))
            if mode == "turn": events.append(dict(kind="page", t=s["start"] + 0.5))
            if mode == "close": events.append(dict(kind="book_close", t=s["start"] + 1.0))
        if ty == "candles":
            cand = []
            if mode == "few":
                for k in range(9):
                    cand.append(dict(x=float(rng.uniform(-2.2, 2.2)), z=float(rng.uniform(3.5, 9)), t=s["start"] + 0.6 + k * (s["end"] - s["start"] - 1.5) / 9))
            elif mode == "three":
                xs = [(-0.9, 3.2), (0.0, 3.9), (0.95, 3.4)]
                for k in range(3):
                    cand.append(dict(x=xs[k][0], z=xs[k][1], t=its[k]["start"] + 0.2, chime=True))
                t0 = its[3]["start"]; t1 = s["end"] - 1.0
                for k in range(160):
                    cand.append(dict(x=float(rng.uniform(-6, 6)), z=float(4.5 + rng.power(0.8) * 22), t=float(t0 + (t1 - t0) * (k / 160) ** 1.6)))
            elif mode == "many":
                for k in range(260):
                    cand.append(dict(x=float(rng.uniform(-7, 7)), z=float(3.0 + rng.power(0.7) * 26), t=float(s["start"] + 0.2 + (s["end"] - s["start"] - 2) * (k / 260) ** 1.3)))
            s["candles"] = cand
            for c in cand[:9 if mode == "few" else 3]:
                events.append(dict(kind="chime", t=c["t"], big=bool(c.get("chime"))))
        if ty == "door" and mode == "closed": events.append(dict(kind="door_appear", t=s["start"] + 0.3))
        if ty == "door" and mode == "open": events.append(dict(kind="door_open", t=its[0]["start"] + 0.2))
        if ty == "hall" and s["params"].get("door") == "appear": events.append(dict(kind="door_appear", t=s["start"] + 0.4))
        if ty == "scale" and mode == "break": events.append(dict(kind="break", t=its[0]["start"] + 0.1))
        for it in its:
            if it["opts"].get("sfx") == "boom": events.append(dict(kind="boom", t=it["start"] + 0.05))
        if ty == "words":
            for it in its:
                if it["spk"] != "S": events.append(dict(kind="tick", t=it["start"]))
        if ty == "eyes": events.append(dict(kind="breath", t=s["start"] + 0.3))

    # ---------- music ----------
    rngm = np.random.default_rng(11)
    mus = np.zeros((2, n), np.float32)
    def note_hz(m): return 440.0 * 2 ** ((m - 69) / 12)
    # intensity by shot type
    inten = np.full(n, 0.55, np.float32)
    dark = np.zeros(n, np.float32)
    for s in shots:
        a, b = int(s["start"] * SR), int(s["end"] * SR)
        v = {"void": 0.35, "eyes": 0.45, "candles": 0.85, "crowd": 0.7, "words": 0.75, "door": 0.9, "end": 0.6}.get(s["type"], 0.55)
        inten[a:b] = v
    verdict_t = [s["start"] for s in shots if s["type"] == "door"][0]
    dark[int(verdict_t * SR):] = 1.0
    k = int(3 * SR); ker = np.ones(k) / k
    inten = movavg(inten, k)
    dark = movavg(dark, 8 * SR)

    chords = [[50, 57, 62, 65, 69], [46, 53, 58, 62, 65], [41, 53, 57, 60, 65], [45, 52, 57, 61, 64],
              [50, 57, 62, 65, 69], [43, 50, 58, 62, 67], [46, 53, 58, 62, 65], [45, 52, 57, 61, 64]]
    seg = 15.0; L = int(seg * SR); fade = int(5 * SR)
    tt = np.arange(L + fade) / SR
    env = np.ones(L + fade, np.float32); env[:fade] = np.linspace(0, 1, fade) ** 1.5; env[-fade:] = np.linspace(1, 0, fade) ** 1.5
    ci = 0; pos = 0
    while pos < n:
        ch = chords[ci % len(chords)]
        blockL = np.zeros(L + fade, np.float32); blockR = np.zeros(L + fade, np.float32)
        for j, m in enumerate(ch):
            f0 = note_hz(m)
            amp = 0.22 if j == 0 else 0.12
            for det, side in ((-0.0018, 0), (0.0021, 1)):
                ph = rngm.uniform(0, 6.28)
                trem = 0.75 + 0.25 * np.sin(2 * np.pi * rngm.uniform(0.05, 0.13) * tt + ph)
                sig = np.zeros_like(tt)
                for h in range(1, 6):
                    sig += np.sin(2 * np.pi * f0 * (1 + det) * h * tt + ph * h) / h ** 1.8
                sig = (sig * trem * amp).astype(np.float32)
                if side == 0: blockL += sig; blockR += sig * 0.6
                else: blockR += sig; blockL += sig * 0.6
        e = min(n, pos + L + fade)
        mus[0, pos:e] += (blockL * env)[: e - pos]; mus[1, pos:e] += (blockR * env)[: e - pos]
        pos += L; ci += 1
    mus[0] = lowpass(mus[0], 900); mus[1] = lowpass(mus[1], 900)
    tn = np.arange(n) / SR
    sub = (np.sin(2 * np.pi * note_hz(26) * tn) * 0.5 + np.sin(2 * np.pi * note_hz(38) * tn) * 0.25).astype(np.float32)
    sub *= (0.35 + 0.65 * dark) * (0.8 + 0.2 * np.sin(2 * np.pi * 0.03 * tn)).astype(np.float32)
    wind = lowpass(rngm.standard_normal(n).astype(np.float32), 350) * 0.9
    wind *= (0.6 + 0.4 * np.sin(2 * np.pi * 0.021 * tn + 1)).astype(np.float32)
    for c in range(2):
        mus[c] = mus[c] * (0.5 + 0.7 * inten) * (1 - 0.35 * dark) + sub * 0.55 + wind * (0.25 + 0.3 * dark)
    # high shimmer during candles
    shimmer = np.zeros(n, np.float32)
    for s in shots:
        if s["type"] == "candles":
            a, b = int(s["start"] * SR), int(s["end"] * SR)
            seglen = b - a; ts = np.arange(seglen) / SR
            w = np.minimum(1, np.minimum(ts / 3, (seglen / SR - ts) / 3))
            shimmer[a:b] += (np.sin(2 * np.pi * note_hz(81) * ts) + 0.6 * np.sin(2 * np.pi * note_hz(88) * ts * 1.001)) * 0.05 * w * (0.6 + 0.4 * np.sin(2 * np.pi * 0.2 * ts))
    mus[0] += shimmer; mus[1] += shimmer * 0.9

    # ---------- sfx ----------
    sfx = np.zeros((2, n), np.float32)
    def add(sig, t0, gain=1.0, pan=0.0):
        i = int(t0 * SR); e = min(n, i + len(sig))
        sfx[0, i:e] += sig[: e - i] * gain * (1 - max(0, pan)); sfx[1, i:e] += sig[: e - i] * gain * (1 + min(0, pan))
    def bell(f, dur=4.0, amp=0.25):
        ts = np.arange(int(dur * SR)) / SR
        s = sum(np.sin(2 * np.pi * f * r * ts) * g * np.exp(-ts * d) for r, g, d in ((1, 1, 1.2), (2.01, 0.35, 2.4), (3.02, 0.12, 3.5), (4.1, 0.06, 5)))
        s *= np.minimum(1, ts / 0.004)
        return (s * amp).astype(np.float32)
    def noise_burst(dur, lo, hi, attack, decay, seed):
        r = np.random.default_rng(seed); m = int(dur * SR); ts = np.arange(m) / SR
        x = bandpass(r.standard_normal(m).astype(np.float32), lo, hi)
        env = np.minimum(1, ts / attack) * np.exp(-np.maximum(0, ts - attack) / decay)
        return (x * env / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)
    def thump(f=70, dur=0.6, amp=0.8):
        ts = np.arange(int(dur * SR)) / SR
        return (np.sin(2 * np.pi * f * ts * (1 - 0.3 * ts)) * np.exp(-ts * 9) * amp).astype(np.float32)
    pent = [note_hz(m) for m in (69, 72, 74, 76, 79, 81)]
    for ei, ev in enumerate(events):
        k_, t0 = ev["kind"], ev["t"]
        if k_ == "page":
            add(noise_burst(0.7, 1200, 7000, 0.18, 0.12, ei) * 0.22, t0, pan=0.2)
        elif k_ == "book_open":
            add(noise_burst(0.9, 600, 5000, 0.3, 0.15, ei) * 0.2, t0); add(thump(90, 0.5, 0.35), t0 + 0.6)
        elif k_ == "book_close":
            add(thump(80, 0.7, 0.7), t0); add(noise_burst(0.25, 300, 3000, 0.005, 0.05, ei) * 0.25, t0)
        elif k_ == "chime":
            f = note_hz(74) if ev.get("big") else pent[ei % len(pent)]
            add(bell(f, 5, 0.16 if ev.get("big") else 0.07), t0, pan=float(np.sin(ei)) * 0.4)
        elif k_ == "door_appear":
            m = int(4 * SR); ts = np.arange(m) / SR
            s = np.sin(2 * np.pi * 55 * ts) * 0.3 + noise_burst(4, 80, 400, 2.0, 1.2, ei) * 0.25
            add((s * np.sin(np.pi * ts / 4)).astype(np.float32) * 0.8, t0)
        elif k_ == "door_open":
            m = int(6 * SR); ts = np.arange(m) / SR
            rum = noise_burst(6, 30, 200, 2.5, 2.5, ei) * 0.8 + np.sin(2 * np.pi * 41 * ts) * 0.4 * np.sin(np.pi * np.minimum(ts / 6, 1))
            fr = 95 + 25 * np.sin(2 * np.pi * 0.7 * ts) + 8 * np.sin(2 * np.pi * 5.3 * ts)
            creak = bandpass((np.mod(np.cumsum(fr) / SR, 1) - 0.5).astype(np.float32), 300, 1800) * np.clip(np.sin(np.pi * ts / 2.4), 0, 1) * (ts < 2.4)
            add((rum + creak * 0.35).astype(np.float32) * 0.7, t0)
        elif k_ == "boom":
            m = int(5 * SR); ts = np.arange(m) / SR
            s = np.sin(2 * np.pi * 36 * ts) * np.exp(-ts / 1.4) + noise_burst(5, 20, 150, 0.02, 1.2, ei) * 0.6
            add(s.astype(np.float32) * 0.9, t0)
        elif k_ == "break":
            add(bell(note_hz(62) * 0.5, 4, 0.12), t0); add(noise_burst(2.5, 2000, 9000, 0.02, 0.8, ei) * 0.08, t0)
        elif k_ == "tick":
            add(bell(note_hz(50), 3, 0.10), t0)
        elif k_ == "breath":
            add(noise_burst(1.6, 300, 2500, 0.6, 0.5, ei) * 0.12, t0)
    sfx_rev = np.stack([fconv(sfx[0], ir_big_L)[:n], fconv(sfx[1], ir_big_R)[:n]])
    sfx = sfx + 0.25 * sfx_rev

    duck = 1 - 0.4 * movavg(venv, 0.4 * SR)
    mus *= 0.16 * duck
    master = np.stack([voiceL, voiceR]) + mus + sfx
    # fades
    fi = int(1.0 * SR); master[:, :fi] *= np.linspace(0, 1, fi)
    end = int((total + 3) * SR); master = master[:, :end]
    fo = int(6 * SR); master[:, -fo:] *= np.linspace(1, 0, fo) ** 2
    peak = np.max(np.abs(master)); master = master / peak * 0.93
    import wave
    pcm = (np.clip(master.T, -1, 1) * 32767).astype("<i2")
    with wave.open("audio.wav", "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
    json.dump(dict(total=end / SR, shots=shots, items=items, events=events), open("timeline.json", "w"), ensure_ascii=False, indent=1)
    print("total", end / SR, "items", len(items), "shots", len(shots))

asyncio.run(main())
