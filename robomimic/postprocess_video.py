"""
Post-process highlight_1080p.mp4 without re-running rollouts.

Changes:
  1. Remove gripper section (always showed false data)
  2. Keep only first 5 success episodes
  3. Add project name + author credits at bottom right
  4. Speed up episode footage 1.5x → total ~90s (1 min 30 sec)
  5. Regenerate end card with correct "5 Successful"
"""

import re, os, sys
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

INPUT    = "runs/highlight_1080p.mp4"
OUTPUT   = "runs/highlight_v2.mp4"
LOG      = "runs/highlight_log.txt"

FPS      = 20
EP_SPEED = 1.5   # 1.5x faster → episodes shrink from ~119s to ~79s; total ≈ 90s
KEEP_N   = 5

VH, VW   = 1080, 1920
LEFT_W   = 420
RIGHT_X  = 1500
RIGHT_W  = VW - RIGHT_X

C_PANEL  = np.array([14,  16,  24],  dtype=np.uint8)
C_BG     = np.array([8,   10,  16],  dtype=np.uint8)
C_DIM    = (55,  62,  78)
C_GRAY   = (130, 138, 155)
C_WHITE  = (230, 235, 245)
C_GREEN  = (40,  220, 100)
C_CYAN   = (0,   210, 255)
C_DIV    = (30,  35,  50)

TITLE_N   = 60
BETWEEN_N = 18
END_N     = 80

# ── Parse log for per-episode frame counts ─────────────────────────────────────
success_frames = []
with open(LOG) as f:
    for line in f:
        m = re.search(r"Composing success #(\d+) \((\d+) frames\)", line)
        if m:
            success_frames.append(int(m.group(2)))

print(f"Found {len(success_frames)} successes: {success_frames}")
keep_frames = success_frames[:KEEP_N]

# ── Build input-frame ranges for each output segment ──────────────────────────
# Structure: [title 60] [ep1] [card 18] [ep2] [card 18] ... [end 80]
segments = []   # (name, in_start, in_end, speed)
pos = 0
segments.append(("title", pos, pos + TITLE_N, 1.0))
pos += TITLE_N

for i, n in enumerate(success_frames):
    if i < KEEP_N:
        segments.append((f"ep{i+1}", pos, pos + n, EP_SPEED))
        segments.append((f"card{i+1}", pos + n, pos + n + BETWEEN_N, 1.0))
    pos += n + BETWEEN_N
# original end card is skipped — we generate a fresh one

print("\nSegments:")
ep_total_in, ep_total_out = 0, 0
for name, s, e, spd in segments:
    out_dur = (e - s) / spd / FPS
    print(f"  {name:8s}  in=[{s:5d},{e:5d}]  speed={spd}x  out={out_dur:.1f}s")
    if name.startswith("ep"):
        ep_total_in  += (e - s)
        ep_total_out += (e - s) / spd

cards_frames = TITLE_N + KEEP_N * BETWEEN_N + END_N
est_total = ep_total_out / FPS + cards_frames / FPS
print(f"\nEstimated total: {est_total:.1f}s ({est_total/60:.2f} min)")

# ── Build sorted list of input frame indices to write ─────────────────────────
to_write = []
for name, start, end, speed in segments:
    fi = float(start)
    while fi < end:
        to_write.append(int(fi))
        fi += speed
to_write = sorted(set(to_write))
print(f"Input frames to sample: {len(to_write)}")

# ── Fonts ─────────────────────────────────────────────────────────────────────
def load_fonts():
    bold = next((p for p in [
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ] if os.path.exists(p)), None)
    def tf(p, sz): return ImageFont.truetype(p, sz) if p else ImageFont.load_default()
    return {
        "hero":  tf(bold, 100),
        "big":   tf(bold, 46),
        "med":   tf(bold, 30),
        "small": tf(bold, 22),
        "tiny":  tf(bold, 16),
        "micro": tf(bold, 13),
    }

fonts = load_fonts()

def tc(d, x, y, text, font, fill):
    bb = d.textbbox((0, 0), text, font=font)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    d.text((x - w//2, y - h//2), text, font=font, fill=fill)

# ── Per-frame modifications ────────────────────────────────────────────────────
def remove_gripper(frame):
    """Overwrite gripper label + bars + CLOSED/OPEN text with panel background."""
    frame[403:567, 0:LEFT_W] = C_PANEL
    return frame

def update_credits(frame):
    """Replace bottom-right single-line credit with project name + authors."""
    rx = RIGHT_X + RIGHT_W // 2   # 1710
    # Clear old text band
    frame[VH - 46:VH, RIGHT_X:VW] = C_PANEL
    pil = Image.fromarray(frame)
    d   = ImageDraw.Draw(pil)
    tc(d, rx, VH - 34, "ENPM690 · UMD  ·  Tactile Guided Insertion",    fonts["micro"], C_DIM)
    tc(d, rx, VH - 17, "Tirth Sadaria  &  Pranav Deshakulkarni Manjunath", fonts["micro"], C_DIM)
    return np.array(pil)

# ── Regenerated end card ───────────────────────────────────────────────────────
def make_end_card():
    frames = []
    for _ in range(END_N):
        c = np.full((VH, VW, 3), C_BG, dtype=np.uint8)
        # Horizontal dividers
        c[VH//2 - 145 : VH//2 - 144, VW//4 : 3*VW//4] = C_DIV
        c[VH//2 + 144 : VH//2 + 145, VW//4 : 3*VW//4] = C_DIV
        # Corner brackets
        col = np.array([0, 100, 140], dtype=np.uint8)
        for x1, x2 in [(60, 130), (VW-130, VW-60)]:
            c[60:62,     x1:x2]  = col
            c[VH-62:VH-60, x1:x2] = col
        for y1, y2 in [(60, 130), (VH-130, VH-60)]:
            c[y1:y2, 60:62]    = col
            c[y1:y2, VW-62:VW-60] = col

        pil = Image.fromarray(c)
        d   = ImageDraw.Draw(pil)
        tc(d, VW//2, VH//2 - 75, f"{KEEP_N} Successful",          fonts["hero"],  C_GREEN)
        tc(d, VW//2, VH//2 + 25, "Demonstrations",                 fonts["big"],   C_WHITE)
        tc(d, VW//2, VH//2 + 85, "FOTS BC-RNN  ·  ENPM690",        fonts["med"],   C_GRAY)
        tc(d, VW//2, VH//2 + 130, "Tactile Guided Insertion",       fonts["small"], C_GRAY)
        tc(d, VW//2, VH//2 + 166,
           "Tirth Sadaria  &  Pranav Deshakulkarni Manjunath",      fonts["tiny"],  C_DIM)
        frames.append(np.array(pil))
    return frames

# ── Stream input → output ──────────────────────────────────────────────────────
reader  = imageio.get_reader(INPUT)
writer  = imageio.get_writer(
    OUTPUT, fps=FPS, macro_block_size=1,
    ffmpeg_params=["-crf", "17", "-preset", "slow", "-pix_fmt", "yuv420p"],
)

write_set = set(to_write)
si = 0          # pointer into sorted to_write
written = 0

print("\nProcessing frames...")
for fi, raw in enumerate(reader):
    if si >= len(to_write):
        break
    if fi == to_write[si]:
        f = np.array(raw).copy()
        f = remove_gripper(f)
        f = update_credits(f)
        writer.append_data(f)
        written += 1
        si += 1
        if written % 200 == 0:
            print(f"  {written} frames written...")

print("Appending new end card...")
for f in make_end_card():
    writer.append_data(f)

writer.close()
reader.close()

total_out = written + END_N
print(f"\nDone! {total_out} frames → {total_out/FPS:.1f}s ({total_out/FPS/60:.2f} min)")
print(f"Output: {OUTPUT}")
