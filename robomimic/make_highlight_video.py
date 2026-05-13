"""
1080p HD highlight reel — successful nut assembly rollouts only.

Layout (1920x1080):
  Center 1080x1080 : agentview LANCZOS-upscaled
  Left/Right 210px : dark HUD side panels
  Header/Footer    : semi-transparent sci-fi readout bars

Memory-efficient: buffers raw 256x256 frames during rollout (not 1920x1080).
Frames are composed into 1080p only after an episode succeeds.

Usage:
    conda activate robomimic_venv
    python make_highlight_video.py [--agent runs/model_epoch_XXXX.pth]
"""

import sys, os, re, argparse
import numpy as np
import torch
import imageio
from PIL import Image, ImageDraw, ImageFont

TACTILE_ROOT = "/media/tirth/Expansion/docker_data_mount/projects/enpm690/Tactile_Guided_Insertion"
sys.path.insert(0, TACTILE_ROOT)
import env_setup  # noqa: F401
from env_setup.make_env import make_fots_env
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

# ── Canvas geometry ────────────────────────────────────────────────────────────
# Three-column layout
#   Left  420px : ALL dials + info (success, step arc, proximity, gripper, EEF)
#   Center 1080px: agentview full height
#   Right  420px : FOTS sensor feeds (wrist cam + tactile L/R stacked)
VW, VH   = 1920, 1080
LEFT_W   = 420
SIM_SIDE = 1080
SIM_X    = LEFT_W                          # 420
RIGHT_X  = SIM_X + SIM_SIDE               # 1500
RIGHT_W  = VW - RIGHT_X                   # 420
HDR_H    = 52
FPS      = 20

# ── Run config ─────────────────────────────────────────────────────────────────
TARGET_SUCCESSES = 10
MAX_ATTEMPTS     = 200
HORIZON          = 800
SEED             = 7

# ── Tactile lowdim (must match training) ───────────────────────────────────────
POOL_H, POOL_W = 8, 8
BLOCK_H = 96  // POOL_H
BLOCK_W = 128 // POOL_W

POLICY_OBS_KEYS = [
    "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
    "object_pos", "object_quat",
    "object_to_robot0_eef_pos", "object_to_robot0_eef_quat",
    "tactile_left_lowdim", "tactile_right_lowdim",
]
KEY_REMAP = {
    "RoundNut_pos":                "object_pos",
    "RoundNut_quat":               "object_quat",
    "RoundNut_to_robot0_eef_pos":  "object_to_robot0_eef_pos",
    "RoundNut_to_robot0_eef_quat": "object_to_robot0_eef_quat",
}

# ── Palette ────────────────────────────────────────────────────────────────────
C_BG      = (8,   10,  16)
C_PANEL   = (14,  16,  24)
C_CYAN    = (0,   210, 255)
C_GREEN   = (40,  220, 100)
C_WHITE   = (230, 235, 245)
C_GRAY    = (130, 138, 155)
C_DIM     = (55,  62,  78)
C_DIV     = (30,  35,  50)
C_BRACKET = (0,   180, 220)

EEF_COLS  = [(220, 80, 80), (80, 220, 80), (80, 130, 255)]  # X=red Y=green Z=blue
EEF_RANGE = [(-0.6, 0.6), (-0.6, 0.6), (0.7, 1.3)]


# ── Fonts ──────────────────────────────────────────────────────────────────────
def load_fonts():
    bold = next((p for p in [
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ] if os.path.exists(p)), None)
    mono = next((p for p in [
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ] if os.path.exists(p)), None)
    def tf(p, sz): return ImageFont.truetype(p, sz) if p else ImageFont.load_default()
    return {
        "hero":   tf(bold, 100),
        "big":    tf(bold, 46),
        "med":    tf(bold, 30),
        "small":  tf(bold, 22),
        "tiny":   tf(bold, 16),
        "mono":   tf(mono, 22),
        "mono_s": tf(mono, 17),
    }


def _wh(d, text, font):
    bb = d.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0], bb[3] - bb[1]

def tc(d, x, y, text, font, fill):
    w, h = _wh(d, text, font)
    d.text((x - w//2, y - h//2), text, font=font, fill=fill)

def tl(d, x, y, text, font, fill):
    d.text((x, y), text, font=font, fill=fill)

def tr(d, x, y, text, font, fill):
    w, _ = _wh(d, text, font)
    d.text((x - w, y), text, font=font, fill=fill)


# ── Obs utilities ──────────────────────────────────────────────────────────────
def tactile_to_lowdim(frame, baseline):
    diff   = frame.astype(np.float32) - baseline.astype(np.float32)
    gray   = diff @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    pooled = gray.reshape(POOL_H, BLOCK_H, POOL_W, BLOCK_W).mean(axis=(1, 3))
    return (pooled / 128.0).ravel().astype(np.float32)

def process_obs(raw_obs, bl, br):
    obs = {KEY_REMAP.get(k, k): v for k, v in raw_obs.items()}
    obs["tactile_left_lowdim"]  = tactile_to_lowdim(raw_obs["tactile_left"], bl)
    obs["tactile_right_lowdim"] = tactile_to_lowdim(raw_obs["tactile_right"], br)
    return {k: obs[k] for k in POLICY_OBS_KEYS if k in obs}

def check_success(env):
    if bool(env._check_success()):
        return True
    inner = env.env
    for i, nut in enumerate(inner.nuts):
        pos = inner.sim.data.body_xpos[inner.obj_body_id[nut.name]]
        if inner.on_peg(pos, i):
            return True
    return False


# ── numpy draw helpers ─────────────────────────────────────────────────────────
def blend(canvas, x1, y1, x2, y2, color, alpha=0.82):
    roi = canvas[y1:y2, x1:x2].astype(np.float32)
    canvas[y1:y2, x1:x2] = (roi * (1 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)

def hline(c, y, x1, x2, col, t=1): c[y:y+t, x1:x2] = col
def vline(c, x, y1, y2, col, t=1): c[y1:y2, x:x+t] = col

def hbar(c, x, y, w, h, frac, col, bg=(20, 24, 34)):
    frac = float(np.clip(frac, 0, 1))
    c[y:y+h, x:x+w] = bg
    if frac > 0:
        c[y:y+h, x:x+int(w*frac)] = col

def vbar(c, x, y, w, h, frac, col, bg=(20, 24, 34)):
    frac = float(np.clip(frac, 0, 1))
    c[y:y+h, x:x+w] = bg
    fill = int(h * frac)
    if fill > 0:
        c[y+h-fill:y+h, x:x+w] = col

def brackets(c, x1, y1, x2, y2, sz=55, t=3, col=C_BRACKET):
    # top-left
    hline(c, y1,    x1,    x1+sz, col, t)
    vline(c, x1,    y1,    y1+sz, col, t)
    # top-right
    hline(c, y1,    x2-sz, x2,    col, t)
    vline(c, x2-t,  y1,    y1+sz, col, t)
    # bottom-left
    hline(c, y2-t,  x1,    x1+sz, col, t)
    vline(c, x1,    y2-sz, y2,    col, t)
    # bottom-right
    hline(c, y2-t,  x2-sz, x2,    col, t)
    vline(c, x2-t,  y2-sz, y2,    col, t)


# ── Proximity arc ──────────────────────────────────────────────────────────────
def prox_arc(canvas, cx, cy, r, frac):
    sz   = r * 2 + 12
    surf = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
    d    = ImageDraw.Draw(surf)
    ox, oy = sz//2, sz//2
    d.arc([ox-r, oy-r, ox+r, oy+r], 0, 360, fill=(40, 45, 60, 160), width=7)
    if frac > 0.01:
        t   = float(np.clip(frac, 0, 1))
        r2g = int(220 * t)
        col = (r2g, int(210 * (1-t)) + 80, int(255 * (1-t)), 230)
        d.arc([ox-r, oy-r, ox+r, oy+r], -90, -90 + t*360, fill=col, width=7)
    d.ellipse([ox-4, oy-4, ox+4, oy+4], fill=(0, 210, 255, 220))

    arr  = np.array(surf)
    rgb  = arr[:, :, :3].astype(np.float32)
    alp  = arr[:, :, 3:4].astype(np.float32) / 255.0
    px, py = cx - sz//2, cy - sz//2
    if py < 0 or px < 0 or py+sz > VH or px+sz > VW:
        return
    roi = canvas[py:py+sz, px:px+sz].astype(np.float32)
    canvas[py:py+sz, px:px+sz] = (roi*(1-alp) + rgb*alp).astype(np.uint8)


# ── Step-progress circular arc (drawn with PIL, composited) ───────────────────
def step_arc(canvas, cx, cy, r, frac, label="STEP"):
    sz   = r * 2 + 14
    surf = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
    d    = ImageDraw.Draw(surf)
    ox, oy = sz//2, sz//2
    d.arc([ox-r, oy-r, ox+r, oy+r], 0, 360, fill=(35, 42, 58, 160), width=9)
    if frac > 0.01:
        d.arc([ox-r, oy-r, ox+r, oy+r], -90, -90 + float(np.clip(frac,0,1))*360,
              fill=(0, 180, 255, 230), width=9)
    arr = np.array(surf)
    rgb = arr[:,:,:3].astype(np.float32)
    alp = arr[:,:,3:4].astype(np.float32) / 255.0
    px, py = cx - sz//2, cy - sz//2
    if py<0 or px<0 or py+sz>VH or px+sz>VW: return
    roi = canvas[py:py+sz, px:px+sz].astype(np.float32)
    canvas[py:py+sz, px:px+sz] = (roi*(1-alp) + rgb*alp).astype(np.uint8)


# ── Main frame composer ────────────────────────────────────────────────────────
# Three-column layout (1920×1080):
#   Left  420px : all dials — success counter, step arc, proximity arc,
#                 gripper bars, EEF X/Y/Z bars, return bar, episode info
#   Center 1080px: agentview upscaled full-height, targeting brackets
#   Right  420px : FOTS sensor feeds — wrist cam, tactile L, tactile R
def compose_frame(fonts, item, success_count, attempt, epoch_label):
    agent  = item["agent"]
    wrist  = item.get("wrist")
    tac_l  = item.get("tac_l")
    tac_r  = item.get("tac_r")
    eef    = item["eef"]
    grip   = item["grip"]
    dist   = item["dist"]
    step   = item["step"]
    ret    = item["ret"]
    ok     = item["success"]

    lx = LEFT_W  // 2   # 210 — center of left panel
    rx = RIGHT_X + RIGHT_W // 2  # 1710 — center of right panel

    # ── Base canvas ───────────────────────────────────────────────────────────
    c = np.full((VH, VW, 3), C_BG, dtype=np.uint8)

    # ── Center: agentview 1080×1080 ──────────────────────────────────────────
    sim = np.array(Image.fromarray(agent[::-1].copy())
                   .resize((SIM_SIDE, SIM_SIDE), Image.LANCZOS))
    c[:, SIM_X:SIM_X + SIM_SIDE] = sim

    # ── Side panel backgrounds ────────────────────────────────────────────────
    c[:, :LEFT_W]   = C_PANEL
    c[:, RIGHT_X:]  = C_PANEL
    vline(c, LEFT_W,     0, VH, C_DIV, 2)
    vline(c, RIGHT_X,    0, VH, C_DIV, 2)

    # ── Header overlay on sim ─────────────────────────────────────────────────
    blend(c, SIM_X, 0, SIM_X+SIM_SIDE, HDR_H, C_BG, 0.82)

    # ── Targeting brackets on sim ─────────────────────────────────────────────
    brackets(c, SIM_X+10, HDR_H+6, SIM_X+SIM_SIDE-10, VH-10, sz=55, t=3)

    # ── Step progress bar at bottom of sim ────────────────────────────────────
    hline(c, VH-5, SIM_X, SIM_X+SIM_SIDE, C_DIM, 5)
    hline(c, VH-5, SIM_X, SIM_X+int(SIM_SIDE*step/HORIZON), (0,180,255), 5)

    # ══════════════════════════════════════════════════════════════════════════
    # LEFT PANEL — dials & data
    # ══════════════════════════════════════════════════════════════════════════

    # 1. Success counter
    # 2. Step arc  (circular progress)
    # 3. Proximity arc
    # 4. Gripper bars
    # 5. EEF X/Y/Z bars
    # 6. Return vertical bar
    # 7. Episode info

    # Step arc (top of left panel)
    step_arc(c, lx, 112, r=70, frac=step/HORIZON)

    # Proximity arc (below step arc)
    prox_frac = float(np.clip(1.0 - dist/0.45, 0, 1))
    prox_arc(c,  lx, 310, r=58, frac=prox_frac)

    # Gripper bars
    g_max = 0.08
    for i in range(2):
        vbar(c, lx - 20 + i*26, 450, 16, 70,
             float(grip[i])/g_max, (0, 200, 255))

    # EEF X/Y/Z horizontal bars
    bx, bw, bh = 18, LEFT_W - 36, 10
    for i, (rng, col) in enumerate(zip(EEF_RANGE, EEF_COLS)):
        frac = (float(eef[i]) - rng[0]) / (rng[1] - rng[0])
        hbar(c, bx, 590 + i*38 + 18, bw, bh, frac, col)

    # Return vertical bar
    vbar(c, lx-12, 780, 24, 160, float(np.clip(ret/100.0, 0, 1)), (50, 215, 100))

    # ══════════════════════════════════════════════════════════════════════════
    # RIGHT PANEL — FOTS sensor feeds
    # ══════════════════════════════════════════════════════════════════════════
    RX1 = RIGHT_X + 14   # left edge of right panel content

    # Wrist cam
    WW = RIGHT_W - 28
    WH = int(WW * 256 / 256)   # square
    if WH > 340: WH = 340
    wy1 = HDR_H + 10
    if wrist is not None:
        wimg = np.array(Image.fromarray(wrist[::-1].copy())
                        .resize((WW, WH), Image.LANCZOS))
        c[wy1:wy1+WH, RX1:RX1+WW] = wimg
    else:
        c[wy1:wy1+WH, RX1:RX1+WW] = (20, 24, 34)
    hline(c, wy1,    RX1, RX1+WW, C_CYAN, 1)
    hline(c, wy1+WH, RX1, RX1+WW, C_CYAN, 1)
    vline(c, RX1,    wy1, wy1+WH,  C_CYAN, 1)
    vline(c, RX1+WW, wy1, wy1+WH,  C_CYAN, 1)

    # Tactile L and R — stacked vertically
    TW = WW
    TH = int(TW * 96 / 128)   # maintain 96:128 aspect
    if TH > 220: TH = 220; TW = int(TH * 128 / 96)

    for idx, tac in enumerate([tac_l, tac_r]):
        ty = wy1 + WH + 30 + idx * (TH + 28)
        tx = RIGHT_X + (RIGHT_W - TW) // 2
        if tac is not None:
            timg = np.array(Image.fromarray(tac[::-1].copy())
                            .resize((TW, TH), Image.LANCZOS))
            c[ty:ty+TH, tx:tx+TW] = timg
        else:
            c[ty:ty+TH, tx:tx+TW] = (20, 24, 34)
        hline(c, ty,    tx, tx+TW, (60,80,110), 1)
        hline(c, ty+TH, tx, tx+TW, (60,80,110), 1)
        vline(c, tx,    ty, ty+TH,  (60,80,110), 1)
        vline(c, tx+TW, ty, ty+TH,  (60,80,110), 1)

    # ── PIL text layer ────────────────────────────────────────────────────────
    pil = Image.fromarray(c)
    d   = ImageDraw.Draw(pil)

    # Sim header
    tc(d, SIM_X + SIM_SIDE//2, HDR_H//2,
       "TACTILE-GUIDED NUT ASSEMBLY",        fonts["small"], C_WHITE)
    tr(d, SIM_X+SIM_SIDE-14, HDR_H//2-10,
       f"EP #{attempt}  EPOCH {epoch_label}", fonts["tiny"],  C_GRAY)

    # ── Left panel text ───────────────────────────────────────────────────────
    tc(d, lx, 16, "FOTS  BC-RNN", fonts["small"], C_CYAN)

    # Step arc label
    tc(d, lx, 42,  "STEP",              fonts["tiny"],  C_GRAY)
    tc(d, lx, 112, f"{step}",           fonts["med"],   C_WHITE)
    tc(d, lx, 150, f"/ {HORIZON}",      fonts["tiny"],  C_DIM)
    tc(d, lx, 175, f"{step/HORIZON*100:.0f}%", fonts["tiny"], (0,180,255))

    d.line([(10, 200), (LEFT_W-10, 200)], fill=C_DIV, width=1)

    # Proximity arc label
    tc(d, lx, 218, "PROXIMITY",         fonts["tiny"],  C_GRAY)
    tc(d, lx, 310, f"{dist:.3f} m",     fonts["mono_s"],C_CYAN)
    tc(d, lx, 378, f"{prox_frac*100:.0f}%", fonts["tiny"], (0,200,200))

    d.line([(10, 405), (LEFT_W-10, 405)], fill=C_DIV, width=1)

    # Gripper
    tc(d, lx, 420, "GRIPPER",           fonts["tiny"],  C_GRAY)
    g_closed = float(np.mean(grip)) < 0.015
    tc(d, lx, 530, "CLOSED" if g_closed else "OPEN",
       fonts["small"], (255,180,0) if g_closed else C_CYAN)

    d.line([(10, 560), (LEFT_W-10, 560)], fill=C_DIV, width=1)

    # EEF labels
    tc(d, lx, 572, "END EFFECTOR",      fonts["tiny"],  C_GRAY)
    for i, (lbl, col) in enumerate(zip(["X", "Y", "Z"], EEF_COLS)):
        y0 = 590 + i*38
        tl(d, bx,      y0, f"EEF {lbl}",         fonts["mono_s"], C_GRAY)
        tr(d, bx+bw,   y0, f"{eef[i]:+.3f} m",   fonts["mono_s"], col)

    d.line([(10, 712), (LEFT_W-10, 712)], fill=C_DIV, width=1)

    # Return bar label
    tc(d, lx, 726, "RETURN",            fonts["tiny"],  C_GRAY)
    tc(d, lx, 952, f"{ret:.1f}",        fonts["mono"],  C_GREEN)

    d.line([(10, 970), (LEFT_W-10, 970)], fill=C_DIV, width=1)

    # Success counter
    tc(d, lx, 1000, str(success_count), fonts["big"],   C_GREEN)
    tc(d, lx, 1042, f"/ {TARGET_SUCCESSES} SUCC", fonts["tiny"], C_GRAY)

    # Progress dots
    dsp = (LEFT_W - 20) / TARGET_SUCCESSES
    for i in range(TARGET_SUCCESSES):
        dx = int(10 + dsp*i + dsp/2)
        d.ellipse([dx-7, 1062-7, dx+7, 1062+7],
                  fill=(C_GREEN if i < success_count else C_DIM))

    # ── Right panel text ──────────────────────────────────────────────────────
    tc(d, rx, 16, "SENSOR FEEDS",       fonts["small"], C_CYAN)

    tc(d, rx, wy1 - 10, "WRIST CAM",    fonts["tiny"],  C_GRAY)

    tac_labels = ["TACTILE  LEFT", "TACTILE  RIGHT"]
    for idx in range(2):
        ty  = wy1 + WH + 30 + idx * (TH + 28)
        lbl_y = ty - 14
        tc(d, rx, lbl_y, tac_labels[idx], fonts["tiny"], C_GRAY)

    # Status badge (bottom of right panel)
    d.line([(RIGHT_X+8, VH-110), (VW-8, VH-110)], fill=C_DIV, width=1)
    if ok:
        tc(d, rx, VH-72, "SUCCESS!",     fonts["big"],  C_GREEN)
        d.rectangle([RIGHT_X+14, VH-94, VW-14, VH-46],
                    outline=C_GREEN, width=2)
    else:
        tc(d, rx, VH-72, "RUNNING",      fonts["med"],  C_GRAY)

    tc(d, rx, VH-24, "ENPM690 · UMD",   fonts["tiny"], C_DIM)

    return np.array(pil)


# ── Splash cards ───────────────────────────────────────────────────────────────
def splash(fonts, lines, n=60):
    c = np.full((VH, VW, 3), C_BG, dtype=np.uint8)
    hline(c, VH//2 - 145, VW//4, 3*VW//4, C_DIV, 1)
    hline(c, VH//2 + 145, VW//4, 3*VW//4, C_DIV, 1)
    # corner brackets on the whole frame
    brackets(c, 60, 60, VW-60, VH-60, sz=70, t=2, col=(0, 100, 140))
    pil = Image.fromarray(c)
    d   = ImageDraw.Draw(pil)
    for dy, text, font, col in lines:
        tc(d, VW//2, VH//2 + dy, text, font, col)
    return [np.array(pil)] * n

def title_card(fonts):
    return splash(fonts, [
        (-75, "FOTS  BC-RNN",                       fonts["hero"], C_CYAN),
        ( 25, "Tactile-Guided Nut Assembly",         fonts["big"],  C_WHITE),
        ( 85, "ENPM690  ·  University of Maryland",  fonts["med"],  C_GRAY),
    ], n=60)

def between_card(fonts, n):
    return splash(fonts, [
        (-40, f"SUCCESS  #{n}",    fonts["hero"], C_GREEN),
        ( 70, "Nut placed on peg", fonts["med"],  C_GRAY),
    ], n=18)

def end_card(fonts):
    return splash(fonts, [
        (-75, f"{TARGET_SUCCESSES} Successful", fonts["hero"], C_GREEN),
        ( 25, "Demonstrations",                  fonts["big"],  C_WHITE),
        ( 85, "FOTS BC-RNN  ·  ENPM690",         fonts["med"],  C_GRAY),
    ], n=80)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent",        default="runs/model_epoch_2800.pth",
                        help="Primary checkpoint")
    parser.add_argument("--agent2",       default="runs/model_epoch_2850.pth",
                        help="Fallback checkpoint (used after --switch_after attempts)")
    parser.add_argument("--switch_after", type=int, default=80,
                        help="Switch to --agent2 after this many attempts if successes < half target")
    parser.add_argument("--output",       default="runs/highlight_1080p.mp4")
    parser.add_argument("--seed",         type=int, default=SEED)
    parser.add_argument("--horizon",      type=int, default=HORIZON)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    fonts  = load_fonts()
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    def load_policy(path):
        m = re.search(r"epoch_(\d+)", path)
        label = m.group(1) if m else os.path.basename(path)
        print(f"Loading policy: epoch {label} ...")
        p, _ = FileUtils.policy_from_checkpoint(ckpt_path=path, device=device, verbose=False)
        return p, label

    policy, epoch_label = load_policy(args.agent)
    policy2_loaded      = False   # lazy-load fallback only if needed

    print("Creating FOTS environment...")
    _cwd = os.getcwd()
    os.chdir(TACTILE_ROOT)
    try:
        env = make_fots_env(
            env_name="NutAssemblySingle", nut_type="round",
            fidelity_mode=True, render_height=96, render_width=128,
            has_offscreen_renderer=True, use_camera_obs=True,
        )
    finally:
        os.chdir(_cwd)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    writer = imageio.get_writer(
        args.output, fps=FPS, macro_block_size=1,
        ffmpeg_params=["-crf", "17", "-preset", "slow", "-pix_fmt", "yuv420p"],
    )
    for f in title_card(fonts):
        writer.append_data(f)

    success_count, attempt = 0, 0

    while success_count < TARGET_SUCCESSES and attempt < MAX_ATTEMPTS:
        # Auto-switch to fallback checkpoint if primary isn't performing
        if (not policy2_loaded
                and attempt >= args.switch_after
                and success_count < TARGET_SUCCESSES // 2
                and os.path.exists(args.agent2)):
            print(f"\n[{attempt} attempts, {success_count} successes] "
                  f"Switching to fallback: {args.agent2}")
            policy, epoch_label = load_policy(args.agent2)
            policy2_loaded = True

        attempt += 1
        policy.start_episode()
        raw_obs = env.reset()
        bl = raw_obs["tactile_left"].copy()
        br = raw_obs["tactile_right"].copy()
        obs = process_obs(raw_obs, bl, br)

        total_return, success = 0.0, False
        buf = []   # lightweight raw-data buffer (NOT 1080p frames)

        for step in range(1, args.horizon + 1):
            act = policy(ob=obs)
            raw_obs, reward, done, _ = env.step(act)
            total_return += reward
            success = check_success(env)
            obs = process_obs(raw_obs, bl, br)

            dist_vec = raw_obs.get("RoundNut_to_robot0_eef_pos",
                       raw_obs.get("SquareNut_to_robot0_eef_pos", np.zeros(3)))
            wf = raw_obs.get("robot0_eye_in_hand_image")
            buf.append({
                "agent": raw_obs["agentview_image"].copy(),
                "wrist": wf.copy() if wf is not None else None,
                "tac_l": raw_obs["tactile_left"].copy(),
                "tac_r": raw_obs["tactile_right"].copy(),
                "eef":   raw_obs["robot0_eef_pos"].copy(),
                "grip":  raw_obs["robot0_gripper_qpos"].copy(),
                "dist":  float(np.linalg.norm(dist_vec)),
                "step":  step,
                "ret":   total_return,
                "success": success,
            })
            if done or success:
                break

        tag = "SUCCESS" if success else "fail"
        print(f"Attempt {attempt:3d} | {tag:7s} | return {total_return:6.1f} | steps {step}")

        if success:
            success_count += 1
            print(f"  -> Composing success #{success_count} ({len(buf)} frames)...")
            for item in buf:
                writer.append_data(compose_frame(fonts, item, success_count, attempt, epoch_label))
            for f in between_card(fonts, success_count):
                writer.append_data(f)

    for f in end_card(fonts):
        writer.append_data(f)
    writer.close()

    print(f"\nDone! {success_count}/{TARGET_SUCCESSES} successes in {attempt} attempts.")
    print(f"Video: {args.output}")


if __name__ == "__main__":
    main()
