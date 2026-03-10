#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 13:33:03 2025

@author: ranabir034
"""
# python crop_segmentation.py --image input.png --out output.png [--debug]

import argparse, sys
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import cv2

# ============================ Data Models ============================
@dataclass
class RowLine:
    p1: Tuple[int, int]
    p2: Tuple[int, int]

@dataclass
class NavResult:
    heading_error_deg: float
    cross_track_error_px: float
    left_line: RowLine
    right_line: RowLine
    nav_line: RowLine
    robot_pos: Tuple[int, int]

# ============================ Utils ============================
def _norm255(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x -= np.nanmin(x)
    m = float(np.nanmax(x))
    if m > 1e-8: x /= m
    return (255.0 * x).clip(0, 255).astype(np.uint8)

def _line_x_at_y(line: Tuple[int,int,int,int], y: int) -> float:
    x1,y1,x2,y2 = line
    dy, dx = (y2 - y1), (x2 - x1)
    if abs(dy) < 1e-9:
        return float(x1)
    t = (y - y1) / (dy + 1e-12)
    return float(x1 + t * dx)

def _extend_to_image(x1,y1,x2,y2,h,w) -> Optional[Tuple[int,int,int,int]]:
    dy, dx = (y2-y1), (x2-x1)
    if abs(dy) < 1e-9:
        return None
    x_top = int(round(x1 + (0 - y1) * (dx/(dy+1e-12))))
    x_bot = int(round(x1 + ((h-1) - y1) * (dx/(dy+1e-12))))
    return (x_top, 0, x_bot, h-1)

def _slope(line: Tuple[int,int,int,int]) -> float:
    x1,y1,x2,y2 = line
    return (x2 - x1) / (y2 - y1 + 1e-12)

def _support(mask: np.ndarray, line: Tuple[int,int,int,int], samples=200, half_width=3) -> float:
    x1,y1,x2,y2 = line
    h, w = mask.shape[:2]
    ys = np.linspace(0, h-1, samples).astype(int)
    good = 0.0
    for y in ys:
        if y2 == y1: continue
        t = (y - y1) / (y2 - y1 + 1e-12)
        x = int(round(x1 + t * (x2 - x1)))
        x0 = max(0, x - half_width); x1b = min(w-1, x + half_width)
        if x0 <= x1b:
            good += (mask[y, x0:x1b+1] > 0).mean()
    return good / max(1, len(ys))

def _valid_pair(L: Tuple[int,int,int,int], R: Tuple[int,int,int,int], h:int, w:int) -> bool:
    mL, mR = _slope(L), _slope(R)
    # Either opposite signs or both small magnitude
    if np.sign(mL) == np.sign(mR) and (abs(mL) > 0.07 and abs(mR) > 0.07):
        return False
    xLb = _line_x_at_y(L, h-1)
    xRb = _line_x_at_y(R, h-1)
    gap = abs(xRb - xLb)
    return (0.05*w) <= gap <= (0.75*w)

# ============================ Masking ============================
def vegetation_mask(bgr: np.ndarray, debug=False, dbg_prefix="") -> np.ndarray:
    h, w = bgr.shape[:2]

    # --- Contrast normalize value channel (helps in harsh lighting)
    hsv0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = cv2.split(hsv0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v_eq = clahe.apply(v_)
    hsv = cv2.merge([h_, s_, v_eq])

    # --- HSV green band (loose)
    lower = np.array([30, 30, 25], dtype=np.uint8)
    upper = np.array([90,255,255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower, upper)

    # --- ExG / ExGR indices
    b, g, r = cv2.split(bgr.astype(np.float32))
    exg  = 2*g - r - b
    exgr = exg - (1.4*r - g)   # = 3G - 2.4R - B
    exg_u8, exgr_u8 = _norm255(exg), _norm255(exgr)

    # --- Otsu + floors (slightly relaxed for pale crops)
    otsu_exg  = int(cv2.threshold(exg_u8,  0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0])
    otsu_exgr = int(cv2.threshold(exgr_u8, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0])
    thr_exg  = max(20,  otsu_exg)
    thr_exgr = max(12, int(0.8*otsu_exgr))
    _, mask_exg  = cv2.threshold(exg_u8,  thr_exg,  255, cv2.THRESH_BINARY)
    _, mask_exgr = cv2.threshold(exgr_u8, thr_exgr, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_and(mask_hsv, cv2.bitwise_or(mask_exg, mask_exgr))

    # --- Morphology (slightly stronger close for broken rows)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)

    # --- Adaptive bottom-ROI (keeps ground rows, adapts to tilt)
    per_row = mask.mean(axis=1)
    if per_row.max() > 0:
        first_strong = np.argmax(per_row > 0.25*per_row.max())
        cutoff = max(int(0.35*h), int(first_strong))
    else:
        cutoff = int(0.40*h)
    roi = np.zeros_like(mask); roi[cutoff:,:] = 255
    mask = cv2.bitwise_and(mask, roi)

    if debug:
        cv2.imwrite(f"{dbg_prefix}mask_hsv.png", mask_hsv)
        cv2.imwrite(f"{dbg_prefix}exg.png", exg_u8)
        cv2.imwrite(f"{dbg_prefix}exgr.png", exgr_u8)
        cv2.imwrite(f"{dbg_prefix}mask.png", mask)
    return mask

# ============================ Pairing ============================
def _pair_best_by_center(lines_ext: List[Tuple[int,int,int,int]],
                         h:int, w:int, mask: Optional[np.ndarray]=None) -> Optional[Tuple[Tuple[int,int,int,int],Tuple[int,int,int,int]]]:
    if not lines_ext: return None
    cands = []
    for ln in lines_ext:
        xb = _line_x_at_y(ln, h-1)
        cands.append((xb, ln))
    cands.sort(key=lambda t: t[0])

    # deduplicate ~10px buckets
    dedup = []
    bucket, acc = None, []
    for xb, ln in cands:
        k = int(round(xb / 10.0))
        if bucket is None or k == bucket:
            bucket = k; acc.append((xb, ln))
        else:
            xb_avg = sum(z[0] for z in acc) / len(acc)
            xt = int(round(sum(_line_x_at_y(z[1], 0)    for z in acc)/len(acc)))
            xb2= int(round(sum(_line_x_at_y(z[1], h-1) for z in acc)/len(acc)))
            dedup.append((xb_avg, (xt,0,xb2,h-1)))
            bucket, acc = k, [(xb, ln)]
    if acc:
        xb_avg = sum(z[0] for z in acc) / len(acc)
        xt = int(round(sum(_line_x_at_y(z[1], 0)    for z in acc)/len(acc)))
        xb2= int(round(sum(_line_x_at_y(z[1], h-1) for z in acc)/len(acc)))
        dedup.append((xb_avg, (xt,0,xb2,h-1)))
    if len(dedup) < 2: return None

    cx = w/2.0
    best, best_score = None, 1e12
    for i in range(len(dedup)-1):
        xb1, L = dedup[i]
        xb2, R = dedup[i+1]
        if not _valid_pair(L, R, h, w):
            continue
        mid = 0.5*(xb1+xb2)
        gap = abs(xb2-xb1)
        score = abs(mid-cx) + 0.01*gap
        if mask is not None:
            sL = _support(mask, L)
            sR = _support(mask, R)
            support_term = 1.0 - 0.5*(sL + sR)  # 0 best, 1 worst
            score += 200.0*support_term
        if score < best_score:
            best_score, best = score, (L, R)
    return best

# ============================ Hough detector ============================
def detect_hough(img: np.ndarray, debug=False, dbg_prefix="",
                 seeds: Optional[Tuple[int,int]]=None) -> Optional[Tuple[RowLine, RowLine]]:
    h, w = img.shape[:2]
    mask = vegetation_mask(img, debug, dbg_prefix)

    # Optional x window if seeds are provided (focus Hough near corridor)
    x_min, x_max = 0, w
    if seeds is not None:
        xl, xr = seeds
        pad = int(0.25*w)
        x_min = max(0, min(xl, xr) - pad)
        x_max = min(w, max(xl, xr) + pad)

    # Auto-Canny thresholds from content
    med = np.median(mask[mask>0]) if (mask>0).any() else 0
    canny_sets = [
        (max(5, int(0.66*med)), min(255, int(1.33*med + 40))),  # adaptive
        (30, 90),
        (60, 180),
    ]

    hough_sets = [
        dict(threshold=60, minLineLength=int(0.20*h), maxLineGap=45),
        dict(threshold=70, minLineLength=int(0.25*h), maxLineGap=30),
        dict(threshold=80, minLineLength=int(0.30*h), maxLineGap=20),
    ]

    angle_tol = 30.0  # allow a bit more tilt

    for ci, (c1,c2) in enumerate(canny_sets):
        # Build edge map emphasizing vertical structure
        sobelx = cv2.Sobel(mask, cv2.CV_32F, 1, 0, ksize=3)
        sobelx = cv2.convertScaleAbs(np.abs(sobelx))
        edges  = cv2.Canny(mask, c1, c2)
        mix    = cv2.addWeighted(edges, 0.6, sobelx, 0.4, 0)

        # restrict to seed window if any
        if x_min>0 or x_max<w:
            mix[:, :x_min] = 0
            mix[:, x_max:] = 0

        if debug:
            cv2.imwrite(f"{dbg_prefix}edges_{ci}.png", edges)
            cv2.imwrite(f"{dbg_prefix}mix_{ci}.png", mix)

        for hi, hs in enumerate(hough_sets):
            linesP = cv2.HoughLinesP(mix, 1, np.pi/180, **hs)
            if linesP is None:
                continue

            lines_ext = []
            for x1,y1,x2,y2 in linesP[:,0,:]:
                dx,dy = x2-x1, y2-y1
                if dx == 0 and dy == 0: 
                    continue
                ang = abs(90.0 - abs(np.degrees(np.arctan2(dy, dx))))
                if ang > angle_tol:
                    continue
                ext = _extend_to_image(x1,y1,x2,y2,h,w)
                if ext is not None:
                    lines_ext.append(ext)

            pair = _pair_best_by_center(lines_ext, h, w, mask=mask)
            if pair:
                L, R = pair
                if debug:
                    dbg = img.copy()
                    for ln in lines_ext:
                        cv2.line(dbg, (ln[0],ln[1]), (ln[2],ln[3]), (120,120,120), 1)
                    cv2.line(dbg, (L[0],L[1]), (L[2],L[3]), (0,255,0), 3)
                    cv2.line(dbg, (R[0],R[1]), (R[2],R[3]), (0,255,255), 3)
                    cv2.imwrite(f"{dbg_prefix}hough_pair_c{ci}_h{hi}.png", dbg)
                return RowLine((L[0],L[1]),(L[2],L[3])), RowLine((R[0],R[1]),(R[2],R[3]))
    return None

# ============================ Histogram fallback ============================
def detect_histogram(img: np.ndarray, debug=False, dbg_prefix="") -> Optional[Tuple[RowLine, RowLine, Tuple[int,int]]]:
    h, w = img.shape[:2]
    mask = vegetation_mask(img, debug, dbg_prefix+"hist_")
    y0 = int(0.45*h); y1 = h-1
    bands = 12
    ys = np.linspace(y0, y1, bands+1).astype(int)
    ptsL, ptsR = [], []

    cx = w/2.0
    for i in range(bands):
        yA, yB = ys[i], ys[i+1]
        band = mask[yA:yB, :]
        colsum = band.sum(axis=0).astype(np.float32)
        colsum = cv2.GaussianBlur(colsum.reshape(1,-1), (1,51), 0).ravel()

        # require non-trivial vegetation
        if colsum.max() < 10 * (yB - yA):
            continue

        left  = colsum[:int(cx)]
        right = colsum[int(cx):]
        xL = int(np.argmax(left)) if left.size>0 else None
        xR = int(np.argmax(right))+int(cx) if right.size>0 else None

        # enforce minimum prominence
        if xL is not None and left.size and left[xL] > 0.3*colsum.max():
            ptsL.append((xL, (yA+yB)//2))
        if xR is not None and right.size and right[xR-int(cx)] > 0.3*colsum.max():
            ptsR.append((xR, (yA+yB)//2))

        if debug:
            vis = np.zeros((120, w, 3), np.uint8)
            s = _norm255(colsum)
            vis[100, :] = 255
            for x in range(w):
                cv2.line(vis, (x,100), (x, 100 - int(80*s[x]/255)), (255,255,255), 1)
            if xL is not None: cv2.line(vis, (xL,0), (xL,119), (0,255,0),1)
            if xR is not None: cv2.line(vis, (xR,0), (xR,119), (0,255,255),1)
            cv2.imwrite(f"{dbg_prefix}hist_band_{i}.png", vis)

    def fit_xy(pts: List[Tuple[int,int]]) -> Optional[Tuple[float,float]]:
        if len(pts) < 4:
            return None
        Y = np.array([p[1] for p in pts], dtype=np.float32)
        X = np.array([p[0] for p in pts], dtype=np.float32)
        A = np.vstack([Y, np.ones_like(Y)]).T
        a, c = np.linalg.lstsq(A, X, rcond=None)[0]
        return float(a), float(c)

    lfit = fit_xy(ptsL)
    rfit = fit_xy(ptsR)
    if lfit is None or rfit is None:
        return None

    aL,cL = lfit; aR,cR = rfit
    lp1 = (int(aL*0 + cL), 0);     lp2 = (int(aL*(h-1) + cL), h-1)
    rp1 = (int(aR*0 + cR), 0);     rp2 = (int(aR*(h-1) + cR), h-1)

    # order by bottom x
    if _line_x_at_y((lp1[0],lp1[1],lp2[0],lp2[1]), h-1) > _line_x_at_y((rp1[0],rp1[1],rp2[0],rp2[1]), h-1):
        lp1,lp2,rp1,rp2 = rp1,rp2,lp1,lp2

    if debug:
        dbg = img.copy()
        cv2.line(dbg, lp1, lp2, (0,255,0), 3)
        cv2.line(dbg, rp1, rp2, (0,255,255), 3)
        cv2.imwrite(f"{dbg_prefix}hist_rows.png", dbg)

    xLb = int(_line_x_at_y((lp1[0],lp1[1],lp2[0],lp2[1]), h-1))
    xRb = int(_line_x_at_y((rp1[0],rp1[1],rp2[0],rp2[1]), h-1))
    return RowLine(lp1,lp2), RowLine(rp1,rp2), (xLb, xRb)

# ============================ Navigation + metrics ============================
def build_nav_and_errors(L: RowLine, R: RowLine, h:int, w:int) -> NavResult:
    xL_top = _line_x_at_y((L.p1[0],L.p1[1],L.p2[0],L.p2[1]), 0)
    xL_bot = _line_x_at_y((L.p1[0],L.p1[1],L.p2[0],L.p2[1]), h-1)
    xR_top = _line_x_at_y((R.p1[0],R.p1[1],R.p2[0],R.p2[1]), 0)
    xR_bot = _line_x_at_y((R.p1[0],R.p1[1],R.p2[0],R.p2[1]), h-1)

    nav_top = (int(0.5*(xL_top+xR_top)), 0)
    nav_bot = (int(0.5*(xL_bot+xR_bot)), h-1)
    nav_line = RowLine(nav_top, nav_bot)

    dx = nav_bot[0] - nav_top[0]
    dy = nav_bot[1] - nav_top[1]
    heading_deg = float(np.degrees(np.arctan2(dx, dy)))

    robot = (int(w/2), h-1)

    t = (robot[1] - nav_top[1]) / (dy if abs(dy)>1e-9 else 1.0)
    t = max(0.0, min(1.0, t))
    nav_x_at_robot = nav_top[0] + t*dx
    cross_px = float(robot[0] - nav_x_at_robot)  # right positive

    return NavResult(heading_deg, cross_px, L, R, nav_line, robot)

# ============================ Drawing ============================
def draw(img: np.ndarray, res: NavResult) -> np.ndarray:
    vis = img.copy()
    cv2.line(vis, res.left_line.p1,  res.left_line.p2,  (0,255,0),   3)
    cv2.line(vis, res.right_line.p1, res.right_line.p2, (0,255,255), 3)
    cv2.line(vis, res.nav_line.p1,   res.nav_line.p2,   (255,0,0),   3)
    cv2.circle(vis, res.robot_pos, 6, (0,0,255), -1)
    cv2.arrowedLine(vis, res.robot_pos, (res.robot_pos[0], max(res.robot_pos[1]-80,0)), (0,0,255), 3, tipLength=0.2)
    cv2.rectangle(vis, (10,10), (10+560, 10+64), (0,0,0), -1)
    cv2.putText(vis, f"Heading error: {res.heading_error_deg:+.2f} deg", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(vis, f"Cross-track: {res.cross_track_error_px:+.1f} px (right=+)", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return vis

# ============================ CLI ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"Could not read image: {args.image}")
    h, w = img.shape[:2]
    prefix = "" if not args.debug else "debug_"

    # 1) Try Hough directly (robust scoring)
    pair = detect_hough(img, debug=args.debug, dbg_prefix=prefix)
    # 2) If fail, do histogram detection, then re-run Hough focused near seeds
    if pair is None:
        hist = detect_histogram(img, debug=args.debug, dbg_prefix=prefix)
        if hist is not None:
            Lh, Rh, seeds = hist
            # try to refine with Hough around histogram corridor
            pair2 = detect_hough(img, debug=args.debug, dbg_prefix=prefix+"seed_", seeds=seeds)
            pair = pair2 if pair2 is not None else (Lh, Rh)

    if pair is None:
        sys.exit("No valid row pair found. Try --debug and tune HSV/ExG thresholds or ROI.")

    L, R = pair
    res = build_nav_and_errors(L, R, h, w)
    vis = draw(img, res)
    if not cv2.imwrite(args.out, vis):
        sys.exit(f"Failed to write: {args.out}")

    print(f"[OK] Saved: {args.out}")
    print(f"Heading error (deg): {res.heading_error_deg:+.3f}")
    print(f"Cross-track (px):    {res.cross_track_error_px:+.3f}")

if __name__ == "__main__":
    main()
