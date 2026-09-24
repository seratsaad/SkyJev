"""Synthetic raw CCD frames for every simulated exposure, FITS I/O, and a quick-look (QL).

`make_frame` builds a raw frame in electrons from the same physics as `obsassist.etc`
(photons_per_A, instrument throughput, extinction, Moffat slit loss and spatial profile),
adds sky (continuum + airglow lines), dark current, hot pixels, a warm column, pixel
response non-uniformity, Poisson noise, cosmic rays and read noise, then converts to ADU
with a bias pedestal (row gradient) and a 32-column overscan strip, clipped to uint16.

Layouts (see `geometry`):
  * long slit (spec, R < 15000: LRIS, DEIMOS, IMACS, LDSS3, MOSFIRE, and FIRE treated as a
    long slit): dispersion along x, 2048 columns spanning lam_min..lam_max with a small
    quadratic term; spatial axis at the native (binned) scale, cropped to <= 400 rows; the
    trace is slightly tilted and curved.
  * echelle (spec, R >= 15000: MIKE, HIRES): 1024 x 1024, orders from m*lam_c = G with
    G = m_ref * lam_ref (so lam_ref sits on a blaze peak), sinc^2 blaze, curved orders with
    spacing ~ m^p (prism cross-disperser for MIKE: wider toward high m).
  * imaging (mode "img"): 1024 x 1024 with the target and a few field stars.

S/N bookkeeping (the frames are smaller than the real detectors): one frame column stands
for r = (A per column at lam_ref) / cfg.dlam_pix native binned pixels, and (HIRES only) one
frame row for q native rows. Every per-pixel quantity is scaled accordingly: source and sky
by the column width in A, dark current by r*q and read noise by sqrt(r*q) (header RNEFF;
RDNOISE is the native value). The S/N per A of a frame therefore equals that of the real
detector, and QL reports S/N per A (and per native pixel = per A * sqrt(cfg.dlam_pix)).
ADU levels are those of an average native pixel: header GAIN = native gain * r * q is the
effective e-/ADU of a frame pixel (GAINNAT the native one) and FULLWELL = native full well
* r * q, so sky/peak levels in ADU and saturation behave like the real detector's.

QL: bias from the overscan (linear in row), gain; object frames: trace found by collapsing
a window around lam_ref (echelle: the order whose blaze peaks at lam_ref), Moffat fit for the
FWHM, sky from the slit on both sides (linear across the slit, clipped), boxcar extraction of
+-h FWHM (the ETC's aperture, h read from etc.compute_rates), variance = sum over the aperture of (counts + RN^2).

Approximations (documented, deliberate):
  * The ETC aperture is reproduced: boxcar of +-h FWHM (h taken from etc.compute_rates, currently
    1.0) with fractional edge pixels whose variance is weighted like their signal (the ETC's npix). The noise of
    the sky estimate is ignored in the S/N (as in the ETC); optimal extraction would do
    ~10-20% better than both.
  * The sky continuum at lam_ref equals sky_mag_ref (the ETC's sky); optical airglow lines
    are added on top (absolute Rayleigh intensities). For NIR configurations the continuum is
    cfg.sky_line_factor x the band average and the OH lines carry the rest, as in the ETC.
  * Throughput is cfg.eff(lam) at the blaze peak; echelle orders are multiplied by the blaze.
  * Line widths: Gaussian LSF with FWHM = lam / R plus the column width in quadrature;
    template features combine intrinsic and instrumental widths analytically (flux/EW kept).
  * The continuum is Source.ab_at (the ETC's SED); templates add only features (lines, bands,
    breaks, Lyman forest), normalised at cfg.lam_ref. Give coloured targets a matching sed
    (e.g. "bb:4700") - a template never re-applies a continuum slope.
  * The spatial profile along the slit is the Moffat's 1-D marginal (as in the ETC); slit
    images are perpendicular to the dispersion (no line tilt); no fringing, bleeding or
    charge-transfer effects; scattered light in echelle frames is a diffuse pedestal of 1 % of
    the (smoothed) column mean.
  * Frames are downsampled: e.g. IMACS f/4 and FIRE are undersampled (lines ~1 column wide).
  * HIRES shows 26 orders (4640-6880 A) with 2x2 spatial downsampling of the 14" decker.
"""

from __future__ import annotations

import warnings
import zlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.special import ndtr

from obsassist.astro.sites import SITES, SUN_ABS_AB, TELESCOPES, Site, Telescope
from obsassist.astro.sky import _interp_band, dark_sky_zenith, delivered_fwhm, extinction_coeff
from obsassist.etc import (
    BETA,
    H_CGS,
    LINEARITY,
    Source,
    _lsf_cum,
    moffat_alpha,
    photons_per_A,
    slit_fraction,
    strip_fraction,
)
from obsassist.instruments.base import InstrumentConfig

__all__ = [
    "FrameRequest",
    "make_frame",
    "quicklook",
    "write_fits",
    "read_fits",
    "geometry",
    "typical_cal_exptime",
    "template_shape",
    "IMAGE_TYPES",
    "TEMPLATES",
]

# ---------------------------------------------------------------------------
# detector / layout constants
# ---------------------------------------------------------------------------
OVERSCAN = 32  # overscan columns appended on the right
BIAS_ADU = 1000.0
ADU_MAX = 65535
LS_NX = 2048  # long-slit spectral columns
LS_MAX_ROWS = 400  # long-slit spatial crop
LS_PAD = 12  # unilluminated rows beyond the ends of short slits
LS_QUAD = -0.02  # quadratic term of the long-slit wavelength solution (fraction)
ECH_NX = 1024
ECH_NY = 1024
ECH_COVER = 1.1  # reference order covers 1.1 free spectral ranges
ECH_QUAD = 0.02
CR_RATE = 3.0e-4  # cosmic-ray events per frame pixel per 1000 s
PRNU = 0.005  # pixel response non-uniformity (rms)
SCATTER = 0.01  # echelle scattered light: fraction of the column mean
FLAT_PEAK_ADU = 30000.0  # at typical_cal_exptime
ARC_PEAK_ADU = 40000.0  # brightest line, at typical_cal_exptime
RAYLEIGH_PH = 1.0e6 / (4.0 * np.pi) / 4.2545e10  # photons s^-1 cm^-2 arcsec^-2 per Rayleigh
C_KMS = 299792.458

IMAGE_TYPES = ("object", "flat", "arc", "bias", "dark", "sky")
TEMPLATES = ("flat", "hot_star", "sn_ia", "sn_ii", "qso", "galaxy", "hii", "m_dwarf", "white_dwarf", "brown_dwarf")

# echelle layouts: m_ref puts lam_ref on the blaze peak of order m_ref; spacing ~ m^p;
# q = spatial downsampling (frame row = q native binned rows)
ECH_SPECS = {
    "MIKE-RED": dict(m_ref=51, p=0.5, q=1, n_orders=None),
    "MIKE-BLUE": dict(m_ref=79, p=0.5, q=1, n_orders=None),
    "HIRES": dict(m_ref=65, p=-0.5, q=2, n_orders=26),
}


@dataclass
class FrameRequest:
    cfg: InstrumentConfig  # the arm's configuration (e.g. get_config("MIKE-RED"), "LRIS-B600")
    image_type: str  # "object" | "flat" | "arc" | "bias" | "dark" | "sky"
    t_exp_s: float
    site: Site
    tel: Telescope
    source: Optional[Source] = None  # object frames
    template: str = "flat"  # continuum/feature template (TEMPLATES)
    z: float = 0.0
    fwhm_arcsec: float = 0.8  # delivered FWHM at cfg.lam_ref
    airmass: float = 1.2
    cloud_mag: float = 0.0  # grey
    sky_mag_ref: float = 21.0  # sky AB mag/arcsec^2 at cfg.lam_ref (Moon/twilight included)
    slit_offset_arcsec: float = 0.0  # centroid offset across the slit
    on_target: bool = True  # False => no source at all
    seed: int = 0
    # on-chip binning as the observer sets it (total, spectral x spatial). (1, 1) or the configuration's
    # own binning (in either axis order, e.g. MIKE (2, 2), HIRES (1, 2) or (2, 1)) = cfg's binning;
    # coarser values bin further. Binning finer than the configuration's is not simulated.
    binning: Tuple[int, int] = (1, 1)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def frame_kind(cfg: InstrumentConfig) -> str:
    if cfg.mode == "img":
        return "imaging"
    return "echelle" if cfg.echelle else "longslit"


def _extra_binning(cfg: InstrumentConfig, binning) -> Tuple[int, int]:
    """Extra (spectral, spatial) binning factors beyond cfg's own, from the total binning an
    observer sets (see FrameRequest.binning)."""
    try:
        a, b = max(1, int(binning[0])), max(1, int(binning[1]))
    except Exception:
        return 1, 1
    own = (cfg.bin_spectral, cfg.bin_spatial)
    if (a, b) in ((1, 1), own, own[::-1]):
        return 1, 1
    return max(1, a // max(own[0], 1)), max(1, b // max(own[1], 1))


def _binned(cfg: InstrumentConfig, binning) -> Tuple[InstrumentConfig, int, int]:
    """cfg with the requested total binning applied, and the extra factors (bx, by)."""
    bx, by = _extra_binning(cfg, binning)
    if (bx, by) == (1, 1):
        return cfg, 1, 1
    return cfg.with_(bin_spectral=cfg.bin_spectral * bx, bin_spatial=cfg.bin_spatial * by), bx, by


def typical_cal_exptime(cfg: InstrumentConfig, image_type: str) -> float:
    """Exposure time (s) giving the nominal calibration level: flats peak at ~30000 ADU,
    the brightest arc line at ~40000 ADU (capped at 60 % / 75 % of the full well for
    low-full-well detectors such as MOSFIRE). Levels scale linearly with t_exp."""
    it = image_type.lower()
    if it == "flat":
        return 10.0
    if it == "arc":
        return 10.0 if frame_kind(cfg) == "echelle" else 3.0
    if it == "bias":
        return 0.0
    return 600.0


def _ro(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    a.flags.writeable = False
    return a


def _telescope_for(cfg: InstrumentConfig) -> Optional[Telescope]:
    for k in cfg.telescopes:
        if k in TELESCOPES:
            return TELESCOPES[k]
    return None


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
_GEO_CACHE: Dict[tuple, dict] = {}


def geometry(cfg: InstrumentConfig, binning=(1, 1)) -> dict:
    """Detector layout and wavelength solution (cached; arrays are read-only).

    Common keys: kind, nx, ny (data section), n_over, pix (arcsec per frame row),
    r / q (native binned pixels per frame column / row), dlam_native (A per native binned
    pixel at lam_ref), lam_ref, slit_px (slit length in frame rows), x_ref, wave, dlam_col,
    ycen (row of the slit centre line / trace shape per column), blaze.
    Long slit: wave/dlam_col/ycen/blaze have shape (1, nx); coeffs (W0, W1, W2) with
    lam(x) = W0 + W1 x + W2 x^2; y_mid (slit centre row); trace_coeffs (T1, T2) of
    ycen = y_mid + T1 (x - x_ref) + T2 (x - x_ref)^2.
    Echelle: arrays have shape (n_orders, nx); orders, G, k1, k2 with
    lam(m, x) = G/m (1 + k1 u + k2 u^2), u = x - (nx-1)/2; i_ref (index of the order
    whose blaze peaks at lam_ref); band_* arrays used to paint the slit images.
    binning: total on-chip binning, as in FrameRequest.binning."""
    cfgb, bx, by = _binned(cfg, binning)
    key = (
        cfg.key,
        cfg.mode,
        cfg.lam_min,
        cfg.lam_max,
        cfg.lam_ref,
        cfg.dispersion,
        cfg.resolution,
        cfg.pixscale,
        cfgb.bin_spatial,
        cfgb.bin_spectral,
        cfg.slit_length,
        cfg.slit_width,
        bx,
        by,
    )
    g = _GEO_CACHE.get(key)
    if g is None:
        kind = frame_kind(cfgb)
        if kind == "echelle":
            g = _geo_echelle(cfgb, bx, by)
        elif kind == "longslit":
            g = _geo_longslit(cfgb, bx, by)
        else:
            g = _geo_imaging(cfgb, bx, by)
        _GEO_CACHE[key] = g
    return g


def _geo_imaging(cfgb, bx, by) -> dict:
    nx, ny = 1024 // bx, 1024 // by
    return dict(
        kind="imaging",
        nx=nx,
        ny=ny,
        n_over=OVERSCAN,
        pix=cfgb.spatial_scale,
        r=1.0,
        q=1.0,
        dlam_native=0.0,
        lam_ref=cfgb.lam_ref,
        slit_px=0.0,
        x_ref=(nx - 1) / 2.0,
        cfg_key=cfgb.key,
    )


def _geo_longslit(cfgb: InstrumentConfig, bx: int, by: int, nx: Optional[int] = None, ny: Optional[int] = None) -> dict:
    nx = int(nx or LS_NX // bx)
    lo, hi = float(cfgb.lam_min), float(cfgb.lam_max)
    n1 = nx - 1
    W2 = LS_QUAD * (hi - lo) / n1**2
    W1 = (hi - lo - W2 * n1**2) / n1
    x = np.arange(nx, dtype=float)
    wave = lo + W1 * x + W2 * x * x
    dl = W1 + 2.0 * W2 * x
    x_ref = float(np.interp(cfgb.lam_ref, wave, x))
    pix = cfgb.spatial_scale
    slit_px = cfgb.slit_length / pix
    if ny is None:
        cap = max(40, LS_MAX_ROWS // by)
        ny = int(min(cap, np.ceil(slit_px) + 2 * LS_PAD))
    y_mid = (ny - 1) / 2.0
    d_ref = float(np.interp(x_ref, x, dl))
    r = d_ref / cfgb.dlam_pix
    T1, T2 = 3.0 / nx, 4.0 / nx**2
    ycen = y_mid + T1 * (x - x_ref) + T2 * (x - x_ref) ** 2
    yy = np.arange(ny, dtype=float)
    s_lo = np.clip((yy - 0.5 - y_mid) * pix, -cfgb.slit_length / 2, cfgb.slit_length / 2)
    s_hi = np.clip((yy + 0.5 - y_mid) * pix, -cfgb.slit_length / 2, cfgb.slit_length / 2)
    s_mid = (yy - y_mid) * pix / (cfgb.slit_length / 2)
    illum = (s_hi - s_lo) / pix * (1.0 - 0.01 * s_mid**2)
    return dict(
        kind="longslit",
        nx=nx,
        ny=ny,
        n_over=OVERSCAN,
        pix=pix,
        r=r,
        q=1.0,
        dlam_native=cfgb.dlam_pix,
        lam_ref=cfgb.lam_ref,
        slit_px=slit_px,
        x_ref=x_ref,
        coeffs=(lo, W1, W2),
        y_mid=y_mid,
        trace_coeffs=(T1, T2),
        wave=_ro(wave[None, :]),
        dlam_col=_ro(dl[None, :]),
        ycen=_ro(ycen[None, :]),
        blaze=_ro(np.ones((1, nx))),
        illum=_ro(illum),
        orders=_ro(np.array([0])),
        i_ref=0,
        cfg_key=cfgb.key,
    )


def _ech_spec(cfgb: InstrumentConfig) -> dict:
    s = ECH_SPECS.get(cfgb.key) or ECH_SPECS.get(cfgb.instrument)
    if s is None:
        s = dict(
            m_ref=max(5, int(round(340000.0 / cfgb.lam_ref))),
            p=0.5,
            q=1 if cfgb.slit_length / cfgb.spatial_scale <= 40 else 2,
            n_orders=None,
        )
    return s


def _geo_echelle(cfgb: InstrumentConfig, bx: int, by: int) -> dict:
    spec = _ech_spec(cfgb)
    nx, ny = ECH_NX // bx, ECH_NY // by
    q = float(spec["q"])
    pix = cfgb.spatial_scale * q
    slit_px = cfgb.slit_length / pix
    lam_ref = float(cfgb.lam_ref)
    m_ref = int(spec["m_ref"])
    G = m_ref * lam_ref
    if spec.get("n_orders"):
        n = int(spec["n_orders"])
        m_lo = m_ref - n // 2
        orders = np.arange(m_lo, m_lo + n)
    else:
        orders = np.arange(int(np.ceil(G / cfgb.lam_max)), int(np.floor(G / cfgb.lam_min)) + 1)
    xc = (nx - 1) / 2.0
    r = max(1.0, ECH_COVER * (lam_ref / m_ref) / (nx * cfgb.dlam_pix))
    k1 = r * cfgb.dlam_pix / lam_ref
    k2 = ECH_QUAD * k1 / xc
    sag, tilt = 12.0 / by, 3.0 / by
    margin = slit_px / 2.0 + 4.0
    p = float(spec["p"])
    while True:
        mid = 0.5 * (orders[:-1] + orders[1:])
        w = (mid / m_ref) ** p
        avail = ny - 2 * margin - 1.2 * sag - 2 * tilt
        s = w * avail / w.sum()
        if s.min() >= slit_px + 3.0 or len(orders) <= 3:
            break
        # drop the order farthest from m_ref
        orders = orders[1:] if abs(orders[0] - m_ref) >= abs(orders[-1] - m_ref) else orders[:-1]
    n_o = len(orders)
    y0 = margin + tilt + np.concatenate([[0.0], np.cumsum(s)])
    x = np.arange(nx, dtype=float)
    u = x - xc
    un = u / xc
    sag_m = sag * (0.8 + 0.4 * np.arange(n_o) / max(n_o - 1, 1))
    ycen = y0[:, None] + sag_m[:, None] * un[None, :] ** 2 + tilt * un[None, :]
    lam_c = G / orders
    wave = lam_c[:, None] * (1.0 + k1 * u[None, :] + k2 * u[None, :] ** 2)
    dl = lam_c[:, None] * (k1 + 2.0 * k2 * u[None, :])
    blaze = np.sinc(orders[:, None] - G / wave) ** 2
    i_ref = int(np.argmin(np.abs(orders - m_ref)))
    # slit-image painting band: rows around each order, fraction of each pixel inside the slit
    half = slit_px / 2.0
    nb = int(np.ceil(2 * half)) + 4
    base = np.floor(ycen - half).astype(int) - 2  # (n_o, nx)
    iy = base[:, None, :] + np.arange(nb)[None, :, None]  # (n_o, nb, nx)
    L2 = cfgb.slit_length / 2.0
    s_lo = np.clip((iy - 0.5 - ycen[:, None, :]) * pix, -L2, L2)
    s_hi = np.clip((iy + 0.5 - ycen[:, None, :]) * pix, -L2, L2)
    s_mid = 0.5 * (s_lo + s_hi) / L2
    frac = (s_hi - s_lo) / pix * (1.0 - 0.01 * s_mid**2)
    inside = (iy >= 0) & (iy < ny)
    frac = np.where(inside, frac, 0.0)
    flat_idx = np.clip(iy, 0, ny - 1) * nx + np.arange(nx)[None, None, :]
    return dict(
        kind="echelle",
        nx=nx,
        ny=ny,
        n_over=OVERSCAN,
        pix=pix,
        r=r,
        q=q,
        dlam_native=cfgb.dlam_pix,
        lam_ref=lam_ref,
        slit_px=slit_px,
        x_ref=xc,
        orders=_ro(orders),
        G=G,
        k1=k1,
        k2=k2,
        p=p,
        m_ref=m_ref,
        i_ref=i_ref,
        wave=_ro(wave),
        dlam_col=_ro(dl),
        ycen=_ro(ycen),
        blaze=_ro(blaze),
        band_frac=_ro(frac),
        band_slo=_ro(s_lo),
        band_shi=_ro(s_hi),
        band_idx=_ro(flat_idx),
        cfg_key=cfgb.key,
    )


_PATTERN_CACHE: Dict[tuple, tuple] = {}


def _fixed_pattern(key: str, ny: int, nx: int):
    """Per-detector fixed pattern: PRNU, hot pixels, a warm column, dead pixels."""
    k = (key, ny, nx)
    if k not in _PATTERN_CACHE:
        rng = np.random.default_rng(zlib.crc32(key.encode()))
        prnu = 1.0 + PRNU * rng.standard_normal((ny, nx))
        n_dead = max(2, int(5e-6 * ny * nx))
        prnu.flat[rng.choice(ny * nx, n_dead, replace=False)] = 0.05
        n_hot = max(4, int(4e-5 * ny * nx))
        hot_idx = rng.choice(ny * nx, n_hot, replace=False)
        hot_rate = 10.0 ** rng.uniform(0.5, 2.3, n_hot)  # e-/s
        bad_x = int(rng.integers(nx // 8, 7 * nx // 8))
        bad_y0 = int(rng.integers(0, max(ny // 2, 1)))
        _PATTERN_CACHE[k] = (_ro(prnu), hot_idx, hot_rate, bad_x, bad_y0)
    return _PATTERN_CACHE[k]


# ---------------------------------------------------------------------------
# spectral templates (f_nu shapes, rest frame; features as (lam_rest, EW [+em/-abs], sigma_rest))
# ---------------------------------------------------------------------------
def _vs(lam, v_kms):
    return lam * v_kms / C_KMS


_BALMER = (6562.8, 4861.3, 4340.5, 4101.7, 3970.1, 3889.1, 3835.4, 3797.9)
_F_IA = 1.0 - 11000.0 / C_KMS
_F_II = 1.0 - 8000.0 / C_KMS

_LINES: Dict[str, list] = {
    "flat": [],
    "hot_star": [
        (lam0, ew, s) for lam0, ew, s in zip(_BALMER, (-4, -5, -5, -5, -4, -3, -2.5, -2), (9, 10, 10, 10, 9, 8, 7, 6))
    ]
    + [(4471.5, -0.8, 3.0), (4026.2, -0.5, 3.0), (3933.7, -0.3, 0.4), (5890.0, -0.25, 0.4), (5895.9, -0.2, 0.4)],
    "white_dwarf": [
        (lam0, ew, s)
        for lam0, ew, s in zip(_BALMER[:7], (-40, -60, -55, -45, -30, -20, -12), (35, 40, 38, 34, 30, 24, 18))
    ],
    "sn_ia": [
        (6355 * _F_IA, -90, 70),
        (5972 * _F_IA, -20, 40),
        (5454 * _F_IA, -25, 35),
        (5640 * _F_IA, -30, 40),
        (3945 * _F_IA, -120, 90),
        (8579 * _F_IA, -150, 120),
        (4900 * _F_IA, -60, 100),
        (4481 * _F_IA, -60, 80),
        (7774 * _F_IA, -40, 70),
        (6355 * 1.005, 45, 90),
        (8579 * 1.005, 70, 140),
        (3945 * 1.01, 40, 90),
    ],
    "sn_ii": [
        (6562.8, 300, 55),
        (6562.8 * _F_II, -50, 45),
        (4861.3, 40, 45),
        (4861.3 * _F_II, -30, 35),
        (5892.9, 30, 50),
        (5892.9 * _F_II, -25, 40),
        (8579.0, 80, 90),
        (8579.0 * _F_II, -60, 80),
        (5169.0 * _F_II, -15, 25),
        (4924.0 * _F_II, -10, 20),
        (5018.0 * _F_II, -12, 22),
    ],
    "qso": [
        (lam0, ew, _vs(lam0, 5000 / 2.355))
        for lam0, ew in (
            (1215.67, 80),
            (1240.1, 15),
            (1400.0, 8),
            (1549.1, 35),
            (1908.7, 20),
            (2798.8, 30),
            (4340.5, 12),
            (4861.3, 60),
            (6562.8, 300),
            (10938.0, 20),
            (12818.0, 30),
            (18751.0, 60),
        )
    ]
    + [(4958.9, 8, _vs(4958.9, 300 / 2.355)), (5006.8, 24, _vs(5006.8, 300 / 2.355))],
    "galaxy": [
        (lam0, ew, _vs(lam0, 200))
        for lam0, ew in (
            (3933.7, -8),
            (3968.5, -7),
            (4304.4, -5),
            (4861.3, -2),
            (5175.4, -6),
            (5269.0, -2.5),
            (5892.9, -4),
            (6562.8, -1.5),
            (8542.1, -3),
            (8662.1, -2.5),
        )
    ]
    + [(lam0, ew, _vs(lam0, 80)) for lam0, ew in ((3727.4, 5), (6562.8, 4), (6583.4, 2), (4861.3, 1))],
    "hii": [
        (lam0, ew, _vs(lam0, 30))
        for lam0, ew in (
            (6562.8, 300),
            (6583.4, 60),
            (6548.1, 20),
            (6716.4, 30),
            (6730.8, 25),
            (4861.3, 70),
            (5006.8, 200),
            (4958.9, 67),
            (3727.4, 150),
            (4340.5, 30),
            (4101.7, 15),
            (3868.8, 20),
            (5875.6, 10),
            (6300.3, 5),
            (7135.8, 8),
            (9068.6, 20),
            (9530.6, 50),
            (10830.0, 30),
            (12818.0, 15),
            (18751.0, 40),
            (21661.0, 2),
        )
    ],
    "m_dwarf": [
        (8183.3, -3, 2),
        (8194.8, -3, 2),
        (7664.9, -2, 2),
        (7699.0, -2, 2),
        (8542.1, -1.5, 2),
        (5892.9, -5, 3),
        (6562.8, 3, 1.5),
    ],
    "brown_dwarf": [(7699.0, -150, 80), (7664.9, -60, 50), (8190.0, -15, 10), (12436.0, -8, 4), (12525.0, -8, 4)],
}
# molecular bands degraded to the red: (head, depth, e-folding scale A)
_BANDS: Dict[str, list] = {
    "m_dwarf": [
        (4954, 0.30, 150),
        (5167, 0.35, 150),
        (5448, 0.35, 150),
        (5847, 0.30, 200),
        (6159, 0.45, 200),
        (6651, 0.30, 150),
        (7054, 0.55, 300),
        (7589, 0.30, 200),
        (8432, 0.25, 200),
        (8859, 0.20, 200),
        (6382, 0.20, 100),
        (6830, 0.25, 100),
    ],
    "brown_dwarf": [(9896, 0.30, 200)],
}


def _bb_fnu(lam, T):
    x = np.clip(1.4388e8 / (lam * T), 1e-6, 700.0)
    return lam**-3.0 / np.expm1(x)


def _soft_box(lr, lo, hi, w):
    return ndtr((lr - lo) / w) * ndtr((hi - lr) / w)


def _structure(name: str, lr) -> np.ndarray:
    """Localised continuum structure of a template (Balmer jump, 4000 A break, UV line
    blanketing, NIR water bands). The smooth continuum slope is NOT part of a template: it is
    Source.sed, exactly as in the ETC, so frames and ETC agree at every wavelength (a red
    'galaxy' should be given e.g. sed="bb:4700"; applying both would count the colour twice)."""
    lr = np.asarray(lr, dtype=float)
    if name in ("flat", "qso", "hii", "m_dwarf"):
        return np.ones_like(lr)
    if name == "hot_star":
        return 0.55 + 0.45 * ndtr((lr - 3646.0) / 12.0)
    if name == "white_dwarf":
        return 0.8 + 0.2 * ndtr((lr - 3646.0) / 12.0)
    if name == "sn_ia":
        return 1.0 / (1.0 + (3900.0 / lr) ** 12)
    if name == "sn_ii":
        return 1.0 / (1.0 + (3300.0 / lr) ** 10)
    if name == "galaxy":
        return (0.45 + 0.55 * ndtr((lr - 4000.0) / 35.0)) / (1.0 + (2800.0 / lr) ** 6)
    if name == "brown_dwarf":
        return (
            1.0
            - 0.35 * _soft_box(lr, 11100.0, 11600.0, 80.0)
            - 0.65 * _soft_box(lr, 13300.0, 14800.0, 150.0)
            - 0.55 * _soft_box(lr, 17800.0, 20200.0, 200.0)
        )
    raise ValueError(f"unknown template {name!r}; known: {TEMPLATES}")


_FOREST_GRID: Dict[str, np.ndarray] = {}


def _lya_forest(lam_obs: np.ndarray, z: float, sigma_obs: float) -> np.ndarray:
    """Ly-alpha / Ly-beta forest transmission (observed frame): a fixed log-normal optical-depth
    field scaled so the mean transmission is exp(-tau_eff), tau_eff = 0.0018 (1+z)^3.92
    (Faucher-Giguere et al. 2008); zero below the Lyman limit; smoothed with the LSF."""
    T = np.ones_like(lam_obs)
    lya = 1215.67 * (1.0 + z)
    m = lam_obs < lya + 20.0
    if not m.any() or z < 1.5:
        return T
    if "g" not in _FOREST_GRID:
        grid = np.arange(2500.0, 30000.0, 0.5)
        rng = np.random.default_rng(1216)
        from scipy.ndimage import gaussian_filter1d

        g = gaussian_filter1d(rng.standard_normal(grid.size), 1.5)
        _FOREST_GRID["x"], _FOREST_GRID["g"] = grid, g / g.std()
    if "A" not in _FOREST_GRID:
        # scale A(tau_eff) such that <exp(-A tau_eff s)> = exp(-tau_eff) over the fixed field
        smp = np.exp(1.2 * _FOREST_GRID["g"][::10] - 0.72)
        te = np.geomspace(0.01, 12.0, 48)
        lo, hi = np.full(te.size, 0.01), np.full(te.size, 1e4)
        for _ in range(50):
            mid = np.sqrt(lo * hi)
            f = np.exp(-np.outer(mid * te, smp)).mean(axis=1)
            big = f < np.exp(-te)
            hi = np.where(big, mid, hi)
            lo = np.where(big, lo, mid)
        _FOREST_GRID["te"], _FOREST_GRID["A"] = te, np.sqrt(lo * hi)
    grid, g = _FOREST_GRID["x"], _FOREST_GRID["g"]
    sel = (grid > lam_obs[m].min() - 60.0) & (grid < lya + 60.0)
    x, gs = grid[sel], g[sel]
    s = np.exp(1.2 * gs - 0.72)

    def tau_of(teff):
        return teff * np.interp(np.log(np.maximum(teff, 1e-3)), np.log(_FOREST_GRID["te"]), _FOREST_GRID["A"]) * s

    za = x / 1215.67 - 1.0
    tau = np.where(x < lya, tau_of(0.0018 * (1.0 + za) ** 3.92), 0.0)
    zb = x / 1025.72 - 1.0
    tau += np.where(x < 1025.72 * (1.0 + z), tau_of(0.0018 * 0.4 * (1.0 + zb) ** 3.92), 0.0)
    tau = np.where(x < 911.8 * (1.0 + z), 30.0, tau)
    tr = np.exp(-tau)
    from scipy.ndimage import gaussian_filter1d

    tr = gaussian_filter1d(tr, max(sigma_obs / 0.5, 0.3))
    T[m] = np.interp(lam_obs[m], x, tr)
    return T


def template_shape(name: str, lam_obs, z: float, sigma_obs, lam_norm: float) -> np.ndarray:
    """Multiplicative spectral features (lines, molecular bands, breaks, Lyman forest) at
    observed wavelengths, normalised to 1 at lam_norm; multiplies Source.ab_at (the continuum).
    sigma_obs: instrumental Gaussian sigma (A, scalar or per point)."""
    lam_obs = np.asarray(lam_obs, dtype=float)
    zp = 1.0 + max(float(z), -0.99)
    lr = lam_obs / zp
    sig_r = np.broadcast_to(np.asarray(sigma_obs, dtype=float), lam_obs.shape) / zp
    out = _structure(name, lr) / float(_structure(name, np.array([lam_norm / zp]))[0])
    acc = np.zeros_like(lam_obs)
    for l0, ew, s0 in _LINES.get(name, []):
        st = np.sqrt(s0**2 + sig_r**2)
        near = np.abs(lr - l0) < 8.0 * st
        if not near.any():
            continue
        acc[near] += ew / (st[near] * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((lr[near] - l0) / st[near]) ** 2)
    mult = np.maximum(1.0 + acc, 0.01)
    for head, depth, scale in _BANDS.get(name, []):
        edge = ndtr((lr - head) / np.maximum(sig_r, 2.0))
        mult = mult * (1.0 - depth * edge * np.exp(-np.maximum(lr - head, 0.0) / scale))
    if name == "qso":
        mult = mult * _lya_forest(lam_obs, float(z), float(np.median(sig_r) * zp))
    return out * mult


# ---------------------------------------------------------------------------
# atmosphere: telluric absorption
# ---------------------------------------------------------------------------
# (lo, hi, mean depth at X=1 for a smooth envelope, line period A, degraded-to-red?)
_TELLURIC = [
    (6866.0, 6910.0, 0.35, 1.8, True),  # O2 B band
    (7160.0, 7340.0, 0.12, 2.0, False),  # H2O
    (7594.0, 7690.0, 0.45, 3.5, True),  # O2 A band
    (8130.0, 8350.0, 0.12, 2.5, False),  # H2O
    (8950.0, 9230.0, 0.10, 2.5, False),  # H2O
    (9300.0, 9800.0, 0.30, 3.0, False),  # H2O
    (11100.0, 11600.0, 0.35, 4.0, False),  # H2O
    (13300.0, 14800.0, 0.45, 5.0, False),  # H2O (between J and H)
    (17900.0, 19600.0, 0.45, 6.0, False),  # H2O (between H and K)
    (20000.0, 20150.0, 0.25, 4.0, False),  # CO2
    (20550.0, 20700.0, 0.25, 4.0, False),  # CO2
]


def telluric(lam, airmass: float, sigma) -> np.ndarray:
    """Telluric transmission. Each band is a soft-edged envelope times a line comb whose
    modulation is damped by the instrumental resolution (exact for a Gaussian LSF acting on
    a sinusoid), so low-resolution spectra see the band average and echelle spectra the
    individual lines. Depth scales with airmass^0.6 (saturated lines)."""
    lam = np.asarray(lam, dtype=float)
    sig = np.broadcast_to(np.asarray(sigma, dtype=float), lam.shape)
    T = np.ones_like(lam)
    for lo, hi, d, P, red in _TELLURIC:
        near = (lam > lo - 30.0) & (lam < hi + 30.0)
        if not near.any():
            continue
        lam_n, s = lam[near], sig[near]
        w = np.maximum(2.0, s)
        env = ndtr((lam_n - lo) / w) * ndtr((hi - lam_n) / w)
        if red:
            env = env * (1.0 - 0.6 * np.clip((lam_n - lo) / (hi - lo), 0.0, 1.0))
        damp = np.exp(-2.0 * np.pi**2 * s**2 / P**2)
        comb = 1.0 + 0.95 * damp * np.cos(2.0 * np.pi * (lam_n - lo) / P)
        t1 = np.clip(1.0 - d * env * comb, 0.0, 1.0)
        T[near] *= t1 ** (max(airmass, 1.0) ** 0.6)
    return T


# ---------------------------------------------------------------------------
# emission-line lists (sky airglow, arc lamps)
# ---------------------------------------------------------------------------
_OH_BANDS = [
    (6257, 0.2),
    (6498, 0.3),
    (6871, 0.6),
    (7276, 0.9),
    (7753, 1.0),
    (7914, 2.2),
    (8344, 3.0),
    (8827, 3.5),
    (9375, 3.0),
    (9788, 2.0),
    (10010, 3.0),
    (10500, 5.0),
    (10880, 6.0),
    (11430, 8.0),
    (12230, 10.0),
    (12900, 12.0),
    (13900, 10.0),
    (14990, 20.0),
    (15600, 25.0),
    (16300, 25.0),
    (16950, 22.0),
    (17700, 15.0),
    (18600, 10.0),
    (19600, 12.0),
    (20650, 8.0),
    (21800, 4.0),
    (22900, 2.5),
    (8645, 0.4),
    (12700, 20.0),
]
_SKY_ATOMIC = [
    (5577.34, 250.0),
    (6300.30, 120.0),
    (6363.78, 40.0),
    (5889.95, 25.0),
    (5895.92, 20.0),
    (4358.34, 3.0),
    (5460.74, 5.0),
    (4046.56, 2.0),
    (5198.5, 10.0),
    (5200.3, 10.0),
    (7319.9, 4.0),
    (7330.2, 3.0),
    (10830.0, 50.0),
]
_SKY_CACHE: Dict[str, tuple] = {}


def _sky_lines() -> Tuple[np.ndarray, np.ndarray]:
    """(wavelength A, intensity Rayleigh) of a synthetic OH Meinel + O2 + atomic airglow list:
    each band has P/Q/R branches of Lambda-doublets in two spin sub-bands."""
    if "sky" not in _SKY_CACHE:
        rng = np.random.default_rng(1729)
        lams, ints = [], []
        for l0, kr in _OH_BANDS:
            dl, it = [], []
            for J in range(1, 8):  # P branch
                dl.append(0.0024 * J + 0.00025 * J * J)
                it.append((2 * J + 1) * np.exp(-J * (J + 1) / 12.0))
            for J in range(1, 4):  # Q branch (compact)
                dl.append(0.0003 * J)
                it.append(0.5 * (2 * J + 1) * np.exp(-J * (J + 1) / 12.0))
            for J in range(1, 4):  # R branch
                dl.append(-(0.0018 * J + 0.0001 * J * J))
                it.append(0.4 * (2 * J + 1) * np.exp(-J * (J + 1) / 12.0))
            dl, it = np.array(dl), np.array(it)
            dl = np.concatenate([dl, dl + 0.0052])
            it = np.concatenate([it, 0.45 * it])  # spin sub-bands
            dl = dl + rng.normal(0, 0.0002, dl.size)
            dl = np.concatenate([dl - 0.00006, dl + 0.00006])
            it = np.concatenate([it, it]) * rng.uniform(0.6, 1.4, dl.size)
            it = it / it.sum() * kr * 1000.0
            lams.append(l0 * (1.0 + dl))
            ints.append(it)
        a = np.array(_SKY_ATOMIC)
        lam = np.concatenate(lams + [a[:, 0]])
        inten = np.concatenate(ints + [a[:, 1]])
        o = np.argsort(lam)
        _SKY_CACHE["sky"] = (_ro(lam[o]), _ro(inten[o]))
    return _SKY_CACHE["sky"]


_HG = [
    (3131.7, 20),
    (3341.5, 5),
    (3650.2, 40),
    (4046.6, 30),
    (4077.8, 5),
    (4358.3, 100),
    (5460.7, 100),
    (5769.6, 15),
    (5790.7, 15),
]
_CDZN = [
    (3261.1, 40),
    (3403.7, 60),
    (3466.2, 60),
    (3610.5, 80),
    (4678.2, 60),
    (4799.9, 80),
    (5085.8, 80),
    (6438.5, 100),
    (3302.6, 40),
    (3345.0, 50),
    (4680.1, 50),
    (4722.2, 60),
    (4810.5, 60),
    (6362.4, 30),
]
_NE = [
    (5852.5, 80),
    (5881.9, 30),
    (5944.8, 40),
    (6030.0, 30),
    (6074.3, 50),
    (6096.2, 40),
    (6143.1, 80),
    (6163.6, 40),
    (6217.3, 40),
    (6266.5, 60),
    (6304.8, 30),
    (6334.4, 60),
    (6383.0, 70),
    (6402.2, 120),
    (6506.5, 80),
    (6532.9, 40),
    (6599.0, 60),
    (6678.3, 60),
    (6717.0, 40),
    (6929.5, 60),
    (7032.4, 80),
    (7173.9, 30),
    (7245.2, 60),
    (7438.9, 40),
    (7488.9, 20),
    (7535.8, 30),
    (8136.4, 20),
    (8300.3, 40),
    (8377.6, 60),
    (8495.4, 40),
    (8591.3, 20),
    (8634.6, 30),
    (8654.4, 60),
    (8780.6, 40),
    (8853.9, 40),
    (9148.7, 30),
    (9201.8, 30),
    (9534.2, 20),
    (9665.4, 40),
    (10844.5, 30),
    (11143.0, 30),
    (11177.5, 40),
    (11390.4, 20),
    (11522.7, 40),
    (11766.8, 30),
    (11984.9, 20),
    (12066.3, 30),
    (12459.4, 20),
    (12689.2, 30),
    (12912.0, 20),
    (13219.0, 20),
    (15230.7, 20),
    (17161.9, 20),
    (18282.6, 20),
]
_AR = [
    (6965.4, 100),
    (7067.2, 80),
    (7147.0, 40),
    (7272.9, 30),
    (7384.0, 60),
    (7503.9, 100),
    (7514.7, 60),
    (7635.1, 100),
    (7723.8, 40),
    (7948.2, 80),
    (8006.2, 50),
    (8014.8, 60),
    (8103.7, 60),
    (8115.3, 150),
    (8264.5, 80),
    (8408.2, 60),
    (8424.6, 70),
    (8521.4, 60),
    (9123.0, 150),
    (9224.5, 60),
    (9657.8, 150),
    (9784.5, 60),
    (10470.1, 60),
    (10673.6, 40),
    (11488.1, 40),
    (12112.3, 50),
    (12456.1, 40),
    (12802.7, 40),
    (12956.7, 50),
    (13214.7, 60),
    (13504.2, 60),
    (13718.6, 40),
    (14093.6, 40),
    (15046.5, 40),
    (16180.1, 30),
    (16940.6, 50),
    (17919.6, 20),
    (20616.2, 30),
    (20986.1, 30),
    (21534.2, 20),
    (22077.1, 20),
    (23133.2, 20),
    (23966.5, 20),
]


def _arc_lines(kind: str, lam_min: float, lam_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """ThAr (echelle) or HgNeAr+CdZn (long slit) lines within [lam_min, lam_max]."""
    key = kind
    if key not in _SKY_CACHE:
        if kind == "ThAr":
            rng = np.random.default_rng(314)
            n = 3800  # density ~ 0.8 (4000/lam)^2 per A
            lam_th = 1.0 / rng.uniform(1.0 / 26000.0, 1.0 / 3000.0, n)
            i_th = 10.0 ** rng.uniform(0.0, 2.5, n)
            ar = np.array(_AR)
            lam = np.concatenate([lam_th, ar[:, 0]])
            inten = np.concatenate([i_th, 5.0 * ar[:, 1]])
        else:
            base = np.array(_HG + _CDZN + _NE + _AR, dtype=float)
            rng = np.random.default_rng(2718)
            # weak filler lines so every range has a usable solution
            lam_f = 1.0 / rng.uniform(1.0 / 26000.0, 1.0 / 3000.0, 500)
            lam = np.concatenate([base[:, 0], lam_f])
            inten = np.concatenate([base[:, 1], rng.uniform(3, 15, 500)])
        o = np.argsort(lam)
        _SKY_CACHE[key] = (lam[o], inten[o])
    lam, inten = _SKY_CACHE[key]
    m = (lam >= lam_min) & (lam <= lam_max)
    return lam[m], inten[m]


def _lines_to_columns(
    wave: np.ndarray, dlc: np.ndarray, line_lam: np.ndarray, line_flux: np.ndarray, sigma_A: np.ndarray
) -> np.ndarray:
    """Pixel-integrated Gaussian lines on a (n_o, nx) column grid. line_flux is in units per
    line (e.g. photons/s/...); returns the same units per column."""
    n_o, nx = wave.shape
    out = np.zeros((n_o, nx))
    if line_lam.size == 0:
        return out
    xs = np.arange(nx, dtype=float)
    for o in range(n_o):
        w = wave[o]
        m = (line_lam > w[0]) & (line_lam < w[-1])
        if not m.any():
            continue
        ll, ff, ss = line_lam[m], line_flux[m], sigma_A[m]
        xl = np.interp(ll, w, xs)
        sp = np.maximum(ss / np.interp(ll, w, dlc[o]), 0.05)
        K = int(min(30, np.ceil(5.0 * sp.max()) + 1))
        cols = np.rint(xl).astype(int)[:, None] + np.arange(-K, K + 1)[None, :]
        frac = ndtr((cols + 0.5 - xl[:, None]) / sp[:, None]) - ndtr((cols - 0.5 - xl[:, None]) / sp[:, None])
        ok = (cols >= 0) & (cols < nx)
        out[o] += np.bincount(cols[ok], weights=(ff[:, None] * frac)[ok], minlength=nx)
    return out


def _sigma_inst(cfg: InstrumentConfig, lam) -> np.ndarray:
    lam = np.asarray(lam, dtype=float)
    if cfg.resolution > 0:
        return lam / (cfg.resolution * 2.3548)
    return np.full_like(lam, 2.0 * cfg.dlam_pix)


def _sky_continuum_mag(site: Site, lam, lam_ref: float, sky_mag_ref: float) -> np.ndarray:
    """Sky continuum AB mag/arcsec^2: the site's dark-sky colour, plus (when sky_mag_ref is
    brighter than dark) a solar-coloured moon/twilight component; equals sky_mag_ref at lam_ref."""
    lam = np.asarray(lam, dtype=float)
    f_dark = 10.0 ** (-0.4 * dark_sky_zenith(site, lam))
    f_dark_ref = float(10.0 ** (-0.4 * dark_sky_zenith(site, lam_ref)))
    f_ref = 10.0 ** (-0.4 * sky_mag_ref)
    if f_ref > 1.05 * f_dark_ref:
        sun = {b: SUN_ABS_AB[b] for b in ("U", "B", "V", "R", "I", "Y", "J", "H", "K")}
        col = 10.0 ** (-0.4 * (_interp_band(sun, lam) - float(_interp_band(sun, lam_ref))))
        f = f_dark + (f_ref - f_dark_ref) * col
    else:
        f = f_dark * (f_ref / f_dark_ref)
    return -2.5 * np.log10(np.maximum(f, 1e-40))


# ---------------------------------------------------------------------------
# frame synthesis
# ---------------------------------------------------------------------------
def _sky_columns(req: FrameRequest, cfg: InstrumentConfig, geo: dict, wave, dlc, tput, t, area) -> np.ndarray:
    """Sky e- per fully illuminated frame pixel, per column (n_o, nx)."""
    factor = cfg.sky_line_factor if cfg.nir else 1.0
    cont = photons_per_A(_sky_continuum_mag(req.site, wave, cfg.lam_ref, req.sky_mag_ref), wave) * factor * dlc
    ll, R = _sky_lines()
    ph = R * RAYLEIGH_PH
    if cfg.nir:
        W = 0.04 * cfg.lam_ref
        win = np.abs(ll - cfg.lam_ref) < W
        tot = float(ph[win].sum())
        n_ref = float(photons_per_A(req.sky_mag_ref, cfg.lam_ref))
        ph = ph * ((1.0 - factor) * n_ref * 2 * W / tot) if tot > 0 else ph * 0.0
    lines = _lines_to_columns(wave, dlc, ll, ph, _sigma_inst(cfg, ll))
    return (cont + lines) * area * tput * cfg.slit_width * geo["pix"] * t


def _fwhm_lambda(req: FrameRequest, cfg: InstrumentConfig, wave) -> np.ndarray:
    """Delivered FWHM at each wavelength as the ETC computes it (etc.compute_rates ->
    sky.delivered_fwhm: atmosphere ~ lam^-0.2 with the von Karman correction, plus the
    telescope floor in quadrature), pinned to req.fwhm_arcsec at cfg.lam_ref."""
    tel = req.tel
    X = max(float(req.airmass), 1.0)
    s = np.geomspace(0.03, 8.0, 400)
    f_ref = delivered_fwhm(s, X, cfg.lam_ref, tel.iq_floor_arcsec, 0.0, tel.outer_scale_m)
    ok = np.all(np.diff(f_ref) > 0) and f_ref[0] < req.fwhm_arcsec < f_ref[-1]
    if not ok:
        return req.fwhm_arcsec * (np.asarray(wave, dtype=float) / cfg.lam_ref) ** -0.2
    seeing = float(np.interp(req.fwhm_arcsec, f_ref, s))
    f = delivered_fwhm(seeing, X, np.asarray(wave, dtype=float), tel.iq_floor_arcsec, 0.0, tel.outer_scale_m)
    f0 = float(delivered_fwhm(seeing, X, cfg.lam_ref, tel.iq_floor_arcsec, 0.0, tel.outer_scale_m))
    return f * (req.fwhm_arcsec / f0)


def _source_columns(req: FrameRequest, cfg: InstrumentConfig, geo: dict, wave, dlc, tput, t, area):
    """Source e- per column entering the slit (point) or per arcsec along the slit (extended)."""
    src = req.source
    sig = np.sqrt(_sigma_inst(cfg, wave) ** 2 + dlc**2 / 12.0)
    # a centroid offset across the slit shifts the source's wavelengths on the detector
    disp = cfg.dispersion * (wave / cfg.lam_ref if geo["kind"] == "echelle" else 1.0)
    ls = wave - req.slit_offset_arcsec / cfg.pixscale * disp
    shape = template_shape(req.template, ls, req.z, sig, cfg.lam_ref)
    m_ab = src.ab_at(ls) - 2.5 * np.log10(np.maximum(shape, 1e-30))
    n = photons_per_A(m_ab, wave)
    k = extinction_coeff(req.site, wave)
    trans = 10.0 ** (-0.4 * (k * req.airmass + req.cloud_mag)) * telluric(ls, req.airmass, sig)
    fwhm = _fwhm_lambda(req, cfg, wave)
    base = n * area * tput * trans * dlc * t
    if src.kind == "point":
        base = base * _slit_frac_fast(cfg.slit_width, fwhm, req.slit_offset_arcsec)
    else:
        base = base * cfg.slit_width
    return base, fwhm


_CDF_TAB: Dict[str, np.ndarray] = {}


def _cdf(a, alpha):
    """Tabulated etc._lsf_cum (Moffat 1-D marginal CDF from 0 to a); it depends only on a/alpha.
    Linear interpolation on a 0.004-step grid: |error| < 5e-6, ~20x faster than betainc."""
    if "u" not in _CDF_TAB:
        u = np.linspace(-80.0, 80.0, 40001)
        _CDF_TAB["u"], _CDF_TAB["f"] = u, _lsf_cum(u, 1.0, BETA)
    return np.interp(np.asarray(a, dtype=float) / alpha, _CDF_TAB["u"], _CDF_TAB["f"])


def _slit_frac_fast(width, fwhm, offset=0.0):
    alpha = moffat_alpha(fwhm)
    return _cdf(width / 2.0 - offset, alpha) - _cdf(-width / 2.0 - offset, alpha)


_AP_HALF: Dict[str, float] = {}


def _etc_ap_half() -> float:
    """Half-width (in FWHM) of the ETC's point-source spectroscopic extraction aperture,
    read from etc.compute_rates so QL and truth follow whatever convention the ETC uses."""
    if "h" not in _AP_HALF:
        try:
            from obsassist.astro.sites import KECK1, MAUNAKEA
            from obsassist.etc import compute_rates

            probe = InstrumentConfig(
                key="_probe",
                instrument="probe",
                telescopes=("keck1",),
                mode="spec",
                description="aperture probe",
                lam_min=5000.0,
                lam_max=8000.0,
                lam_ref=6500.0,
                throughput=((5000.0, 0.3), (8000.0, 0.3)),
                pixscale=0.01,
                read_noise=1.0,
                dark_e_per_hr=0.0,
                gain=1.0,
                full_well=1e5,
                readout_s=1.0,
                dispersion=1.0,
            )
            r = compute_rates(probe, KECK1, MAUNAKEA, Source(20.0), seeing_500=1.0, airmass=1.0, sky_mag=21.0)
            h = float(r.npix) * probe.spatial_scale / (2.0 * float(r.fwhm))
            _AP_HALF["h"] = h if 0.2 < h < 5.0 else 1.0
        except Exception:
            _AP_HALF["h"] = 1.0
    return _AP_HALF["h"]


def _moffat_cdf_full(s, alpha):
    return 0.5 + _cdf(s, alpha)


def _paint_source_longslit(img, geo, cfg, col_e, fwhm, src: Source):
    ny, nx = img.shape
    pix, y_mid = geo["pix"], geo["y_mid"]
    ycen = geo["ycen"][0]
    ext = src.kind != "point"
    L_ext = float(src.extent_arcsec) if ext else 0.0
    K = int(np.ceil((8.0 * float(fwhm.max()) + L_ext / 2) / pix)) + 3
    y_lo = max(0, int(np.floor(ycen.min())) - K)
    y_hi = min(ny, int(np.ceil(ycen.max())) + K + 1)
    if y_hi <= y_lo:
        return
    L2 = cfg.slit_length / 2.0
    s_edge = np.clip((np.arange(y_lo, y_hi + 1) - 0.5 - y_mid) * pix, -L2, L2)  # (nb+1,)
    rel = s_edge[:, None] - ((ycen - y_mid) * pix)[None, :]  # (nb+1, nx)
    alpha = moffat_alpha(fwhm)[None, :]
    if not ext:
        prof = np.diff(_cdf(rel, alpha), axis=0)
    else:
        mid = 0.5 * (rel[1:] + rel[:-1])
        prof = (_moffat_cdf_full(mid + L_ext / 2, alpha) - _moffat_cdf_full(mid - L_ext / 2, alpha)) * np.diff(
            rel, axis=0
        )
    img[y_lo:y_hi] += prof * col_e[None, :]


def _source_profile_echelle(geo, fwhm, src: Source):
    slo, shi = geo["band_slo"], geo["band_shi"]
    alpha = moffat_alpha(fwhm)[:, None, :]
    if src.kind == "point":
        edges = np.concatenate([slo, shi[:, -1:, :]], axis=1)
        return np.diff(_cdf(edges, alpha), axis=1)
    L = float(src.extent_arcsec)
    mid = 0.5 * (slo + shi)
    return (_moffat_cdf_full(mid + L / 2, alpha) - _moffat_cdf_full(mid - L / 2, alpha)) * (shi - slo)


def _add_cosmics(e: np.ndarray, t: float, rng) -> int:
    ny, nx = e.shape
    n = int(rng.poisson(CR_RATE * ny * nx * t / 1000.0)) if t > 0 else 0
    if n == 0:
        return 0
    y0, x0 = rng.uniform(0, ny, n), rng.uniform(0, nx, n)
    L = rng.integers(1, 7, n)
    ang = rng.uniform(0, np.pi, n)
    E = 10.0 ** rng.uniform(np.log10(2000.0), np.log10(40000.0), n)
    rep = np.repeat(np.arange(n), L)
    j = np.arange(L.sum()) - np.repeat(np.cumsum(L) - L, L)
    ys = np.floor(y0[rep] + j * np.sin(ang[rep])).astype(int)
    xs = np.floor(x0[rep] + j * np.cos(ang[rep])).astype(int)
    w = E[rep] / L[rep] * rng.uniform(0.6, 1.4, rep.size)
    ok = (ys >= 0) & (ys < ny) & (xs >= 0) & (xs < nx)
    np.add.at(e, (ys[ok], xs[ok]), w[ok])
    return n


def _detector(img: np.ndarray, cfg: InstrumentConfig, geo: dict, t: float, rng) -> Tuple[np.ndarray, int, float]:
    """Fixed pattern, dark, Poisson, cosmic rays, full well, bias + read noise, overscan."""
    ny, nx = img.shape
    prnu, hot_idx, hot_rate, bad_x, bad_y0 = _fixed_pattern(cfg.key, ny, nx)
    r, q = geo["r"], geo["q"]
    img = img * prnu
    img += cfg.dark_e_per_hr / 3600.0 * cfg.bin_spatial * cfg.bin_spectral * r * q * t
    if t > 0:
        img.flat[hot_idx] += hot_rate * t
        img[bad_y0:, bad_x] += 0.5 * t + 15.0
    e = rng.poisson(np.maximum(img, 0.0)).astype(float)
    n_cr = _add_cosmics(e, t, rng)
    np.minimum(e, cfg.full_well * r * q, out=e)
    rn_eff = cfg.read_noise * np.sqrt(r * q)
    g_eff = cfg.gain * r * q
    bias_row = BIAS_ADU + rng.normal(0.0, 1.5) + 4.0 * np.arange(ny) / ny
    full = np.empty((ny, nx + OVERSCAN))
    full[:, :nx] = e / g_eff
    full[:, nx:] = 0.0
    full += bias_row[:, None] + rng.normal(0.0, rn_eff / g_eff, full.shape)
    np.rint(full, out=full)
    np.clip(full, 0, ADU_MAX, out=full)
    return full.astype(np.uint16), n_cr, rn_eff


def _header(req: FrameRequest, cfg: InstrumentConfig, geo: dict, itype: str, t: float, rn_eff: float) -> dict:
    ny, nx = geo["ny"], geo["nx"]
    rq = geo["r"] * geo["q"]
    h = {
        "INSTRUME": cfg.instrument,
        "INSTCFG": cfg.key,
        "TELESCOP": req.tel.name,
        "OBSTYPE": itype.upper(),
        "EXPTIME": float(t),
        "NAXIS1": nx + OVERSCAN,
        "NAXIS2": ny,
        "FRMTYPE": geo["kind"].upper(),
        "BINNING": f"{cfg.bin_spectral},{cfg.bin_spatial}",
        "GAIN": float(cfg.gain * rq),
        "GAINNAT": float(cfg.gain),
        "RDNOISE": float(cfg.read_noise),
        "RNEFF": float(rn_eff),
        "DARKCUR": float(cfg.dark_e_per_hr),
        "FULLWELL": float(cfg.full_well * rq),
        "FWNAT": float(cfg.full_well),
        "SATURATE": float(min(ADU_MAX, BIAS_ADU + cfg.full_well / cfg.gain)),
        "BIASLVL": BIAS_ADU,
        "OVSCAN": OVERSCAN,
        "DATASEC": f"[1:{nx},1:{ny}]",
        "BIASSEC": f"[{nx + 1}:{nx + OVERSCAN},1:{ny}]",
        "PIXSCALE": float(geo["pix"]),
        "LAMREF": float(cfg.lam_ref),
        "AIRMASS": float(req.airmass),
    }
    if geo["kind"] == "imaging":
        h.update({"FILTER": cfg.band or "", "BANDWID": float(cfg.band_width)})
        return h
    h.update(
        {
            "DISPAXIS": 1,
            "SLITWID": float(cfg.slit_width),
            "SLITLEN": float(cfg.slit_length),
            "RESOLUT": float(cfg.resolution),
            "NATPIXX": float(geo["r"]),
            "NATPIXY": float(geo["q"]),
            "DNATIVE": float(cfg.dlam_pix),
        }
    )
    if geo["kind"] == "longslit":
        W0, W1, W2 = geo["coeffs"]
        T1, T2 = geo["trace_coeffs"]
        h.update(
            {
                "WAVE0": float(W0),
                "DWAVE": float(W1),
                "DWAVE2": float(W2),
                "WAVEUNIT": "Angstrom",
                "TRCA1": float(T1),
                "TRCA2": float(T2),
                "TRCXREF": float(geo["x_ref"]),
                "SLITROW": float(geo["y_mid"]),
            }
        )
    else:
        o = geo["orders"]
        h.update(
            {
                "ECHG": float(geo["G"]),
                "ECHK1": float(geo["k1"]),
                "ECHK2": float(geo["k2"]),
                "ORDLO": int(o.min()),
                "ORDHI": int(o.max()),
                "NORDERS": int(o.size),
                "ORDREF": int(geo["m_ref"]),
                "ORDXREF": float(geo["x_ref"]),
                "ORDSPP": float(geo["p"]),
            }
        )
    return h


def _truth_ref(req: FrameRequest, cfg: InstrumentConfig, t: float, area: float, has_source: bool) -> dict:
    """Noiseless expectations at lam_ref in native (binned) pixels, as the ETC defines them."""
    lam = cfg.lam_ref
    eta = float(cfg.eff(lam))
    dl = cfg.dlam_pix
    fwhm = float(req.fwhm_arcsec)
    trans = float(10.0 ** (-0.4 * (float(extinction_coeff(req.site, lam)) * req.airmass + req.cloud_mag)))
    factor = cfg.sky_line_factor if cfg.nir else 1.0
    sky = float(photons_per_A(req.sky_mag_ref, lam)) * area * eta * dl * cfg.slit_width * cfg.spatial_scale * t * factor
    dark = cfg.dark_e_per_hr / 3600.0 * cfg.bin_spatial * cfg.bin_spectral * t
    out = dict(
        lam_ref=lam,
        fwhm_ref=fwhm,
        transmission=trans,
        sky_mag_ref=req.sky_mag_ref,
        sky_e_native_pix=sky,
        dark_e_native_pix=dark,
        read_noise=cfg.read_noise,
        dlam_native=dl,
    )
    if has_source and req.source.kind == "point":
        src = req.source
        m = float(src.ab_at(lam))
        fsl = float(slit_fraction(cfg.slit_width, fwhm, req.slit_offset_arcsec))
        hap = _etc_ap_half() * fwhm
        fext = float(strip_fraction(-hap, hap, fwhm))
        sig = float(photons_per_A(m, lam)) * area * eta * trans * dl * fsl * t
        S = sig * fext
        npix = max(2.0 * hap / cfg.spatial_scale, 1.0)
        var = S + npix * (sky + dark + cfg.read_noise**2)
        snr_pix = S / np.sqrt(max(var, 1e-12))
        out.update(
            mag_ref=m,
            slit_frac=fsl,
            ap_frac=fext,
            signal_e_native=sig,
            signal_e_ap_native=S,
            snr_ref_per_pix=float(snr_pix),
            snr_ref_per_A=float(snr_pix / np.sqrt(dl)),
        )
    return out


def make_frame(req: FrameRequest) -> Tuple[np.ndarray, dict, dict]:
    """(data uint16 [ny, nx], header_extras, truth); truth holds the noiseless source/sky
    electrons at lam_ref (native pixels), the expected S/N, the true trace row, etc."""
    itype = str(req.image_type).lower().strip()
    if itype not in IMAGE_TYPES:
        raise ValueError(f"image_type must be one of {IMAGE_TYPES}, got {req.image_type!r}")
    if req.template not in TEMPLATES:
        raise ValueError(f"unknown template {req.template!r}; known: {TEMPLATES}")
    cfg, bx, by = _binned(req.cfg, req.binning)
    t = 0.0 if itype == "bias" else max(float(req.t_exp_s), 0.0)
    rng = np.random.default_rng(req.seed)
    area = req.tel.area_m2 * 1e4
    geo = geometry(req.cfg, req.binning)
    has_source = itype == "object" and bool(req.on_target) and req.source is not None
    if geo["kind"] == "imaging":
        img = _image_imaging(req, cfg, geo, itype, t, area, has_source)
    else:
        img = _image_spectrum(req, cfg, geo, itype, t, area, has_source)
    data, n_cr, rn_eff = _detector(img, cfg, geo, t, rng)
    header = _header(req, cfg, geo, itype, t, rn_eff)
    truth = dict(
        image_type=itype,
        kind=geo["kind"],
        template=req.template,
        z=req.z,
        on_target=bool(req.on_target),
        has_source=has_source,
        r=geo["r"],
        q=geo["q"],
        t_exp=t,
        n_cosmic=n_cr,
        rn_eff=rn_eff,
    )
    if geo["kind"] != "imaging":
        truth.update(_truth_ref(req, cfg, t, area, has_source))
        if geo["kind"] == "longslit":
            truth["trace_row"] = float(geo["y_mid"]) if has_source else None
            truth["x_ref"] = float(geo["x_ref"])
        else:
            i, xr = geo["i_ref"], int(round(geo["x_ref"]))
            truth["trace_row"] = float(geo["ycen"][i, xr]) if has_source else None
            truth["order_ref"] = int(geo["orders"][i])
            truth["x_ref"] = float(geo["x_ref"])
    return data, header, truth


def _image_spectrum(req, cfg, geo, itype, t, area, has_source) -> np.ndarray:
    ny, nx = geo["ny"], geo["nx"]
    wave, dlc = geo["wave"], geo["dlam_col"]
    tput = cfg.eff(wave) * geo["blaze"]
    slit_col = np.zeros_like(wave)
    src_col = fwhm = None
    if itype in ("object", "sky"):
        slit_col += _sky_columns(req, cfg, geo, wave, dlc, tput, t, area)
        if has_source:
            src_col, fwhm = _source_columns(req, cfg, geo, wave, dlc, tput, t, area)
    elif itype == "flat":
        lamp = _bb_fnu(wave, 3100.0) / wave * dlc * tput  # quartz photons per column
        t0 = typical_cal_exptime(cfg, "flat")
        peak_adu = min(FLAT_PEAK_ADU, 0.6 * cfg.full_well / cfg.gain)
        slit_col += lamp * (peak_adu * cfg.gain * geo["r"] * geo["q"] * t / t0 / max(float(lamp.max()), 1e-300))
    elif itype == "arc":
        echelle = geo["kind"] == "echelle"
        ll, inten = _arc_lines("ThAr" if echelle else "HgNeAr", float(wave.min()), float(wave.max()))
        spec = _lines_to_columns(wave, dlc, ll, inten, _sigma_inst(cfg, ll)) * tput
        t0 = typical_cal_exptime(cfg, "arc")
        peak_adu = min(ARC_PEAK_ADU, 0.75 * cfg.full_well / cfg.gain)
        slit_col += spec * (peak_adu * cfg.gain * geo["r"] * geo["q"] * t / t0 / max(float(spec.max()), 1e-300))
    if geo["kind"] == "longslit":
        img = geo["illum"][:, None] * slit_col[0][None, :]
        if src_col is not None:
            _paint_source_longslit(img, geo, cfg, src_col[0], fwhm[0], req.source)
    else:
        vals = geo["band_frac"] * slit_col[:, None, :]
        if src_col is not None:
            vals = vals + _source_profile_echelle(geo, fwhm, req.source) * src_col[:, None, :]
        img = np.bincount(geo["band_idx"].ravel(), vals.ravel(), minlength=ny * nx).reshape(ny, nx)
        if itype in ("object", "sky", "flat", "arc"):  # diffuse scattered light
            from scipy.ndimage import gaussian_filter1d

            img += SCATTER * gaussian_filter1d(img.mean(axis=0), 60.0, mode="nearest")[None, :]
    return img


def _image_imaging(req, cfg, geo, itype, t, area, has_source) -> np.ndarray:
    ny, nx, pix = geo["ny"], geo["nx"], geo["pix"]
    eta = float(cfg.eff(cfg.lam_ref))
    bw = cfg.band_width or 1000.0
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr2 = ((yy - ny / 2) ** 2 + (xx - nx / 2) ** 2) / (0.5 * max(nx, ny)) ** 2
    vign = 1.0 - 0.03 * rr2
    img = np.zeros((ny, nx))
    if itype in ("object", "sky"):
        img += float(photons_per_A(req.sky_mag_ref, cfg.lam_ref)) * area * eta * bw * pix**2 * t * vign
    elif itype == "flat":
        img += (
            min(FLAT_PEAK_ADU, 0.6 * cfg.full_well / cfg.gain) * cfg.gain * t / typical_cal_exptime(cfg, "flat") * vign
        )
    if has_source:
        trans = float(10.0 ** (-0.4 * (float(extinction_coeff(req.site, cfg.lam_ref)) * req.airmass + req.cloud_mag)))
        tot = float(photons_per_A(req.source.ab_at(cfg.lam_ref), cfg.lam_ref)) * area * eta * bw * trans * t
        fw_px = req.fwhm_arcsec / pix
        frng = np.random.default_rng(zlib.crc32(f"{cfg.key}:{req.source.mag:.2f}".encode()))
        stars = [((ny - 1) / 2.0 + 0.3, (nx - 1) / 2.0 + req.slit_offset_arcsec / pix + 0.2, tot)]
        for _ in range(12):
            stars.append(
                (frng.uniform(20, ny - 20), frng.uniform(20, nx - 20), tot * 10 ** (-0.4 * frng.uniform(0.5, 6.0)))
            )
        for yc, xc, f in stars:
            _add_moffat_stamp(img, yc, xc, fw_px, f)
    return img


def _add_moffat_stamp(img, yc, xc, fwhm_px, flux):
    ny, nx = img.shape
    K = int(np.ceil(6 * fwhm_px)) + 2
    y0, x0 = int(round(yc)), int(round(xc))
    ys = np.arange(max(0, y0 - K), min(ny, y0 + K + 1))
    xs = np.arange(max(0, x0 - K), min(nx, x0 + K + 1))
    if ys.size == 0 or xs.size == 0:
        return
    alpha = float(moffat_alpha(fwhm_px))
    sub = (np.arange(3) - 1) / 3.0
    dy = ys[:, None, None, None] + sub[None, None, :, None] - yc
    dx = xs[None, :, None, None] + sub[None, None, None, :] - xc
    prof = (1.0 + (dy**2 + dx**2) / alpha**2) ** -BETA
    prof = prof.mean(axis=(2, 3)) * (BETA - 1.0) / (np.pi * alpha**2)
    img[ys[0] : ys[-1] + 1, xs[0] : xs[-1] + 1] += flux * prof


# ---------------------------------------------------------------------------
# quick-look
# ---------------------------------------------------------------------------
_TYPE_ALIASES = {
    "science": "object",
    "obj": "object",
    "target": "object",
    "std": "object",
    "standard": "object",
    "zero": "bias",
    "domeflat": "flat",
    "lampflat": "flat",
    "quartz": "flat",
    "milky": "flat",
    "twiflat": "sky",
    "comp": "arc",
    "comparison": "arc",
    "lamp": "arc",
    "thar": "arc",
}


def _ql_default() -> dict:
    return {
        "image_type": "unknown",
        "frame_kind": None,
        "trace_found": False,
        "trace_pos": None,
        "fwhm_arcsec": None,
        "peak_adu": None,
        "peak_adu_raw": None,
        "saturated_frac": 0.0,
        "sky_adu_per_pix": None,
        "counts_ref_e": None,
        "snr_ref_per_pix": 0.0,
        "snr_ref_per_A": 0.0,
        "wave": [],
        "flux": [],
        "sky": [],
        "snr_curve": [],
        "cosmic_rays": 0,
        "flags": [],
        "median_adu": None,
        "sky_mag_ref": None,
        "mag_ref_est": None,
        "order_ref": None,
        "units": {
            "wave": "Angstrom",
            "flux": "e-/A in this exposure (echelle: blaze-corrected)",
            "sky": "sky e-/A in the extraction aperture",
            "snr_curve": "S/N per A",
            "snr_ref_per_pix": "S/N per native (binned) detector pixel at lam_ref",
            "counts_ref_e": "source e- per native pixel at lam_ref in the aperture",
            "sky_adu_per_pix": "sky ADU per frame pixel at the trace, lam_ref",
            "trace_pos": "row (0-based) of the trace at lam_ref",
        },
    }


def _hget(hdr: dict, key: str, default=None, cast=float):
    v = hdr.get(key)
    if v is None or (isinstance(v, str) and not v.strip()):
        return default
    try:
        out = cast(v)
    except (TypeError, ValueError):
        return default
    if isinstance(out, float) and not np.isfinite(out):
        return default
    return out


def quicklook(data: np.ndarray, header: dict, cfg: InstrumentConfig) -> dict:
    """Quick-look analysis of a raw frame: a pure function of the pixels + header (+ the
    deterministic geometry model of cfg). Never raises; problems are reported in "flags"."""
    res = _ql_default()
    try:
        _ql_run(data, header, cfg, res)
    except Exception as exc:  # pragma: no cover - defensive
        res["flags"].append(f"quick-look failed ({type(exc).__name__}: {exc})")
    return _py(res)


def _py(x):
    """numpy scalars/arrays -> plain Python (JSON-friendly); non-finite floats -> None."""
    if isinstance(x, dict):
        return {k: _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, np.ndarray):
        return _py(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if np.isfinite(x) else None
    return x


def _cr_mask(img: np.ndarray, rn: float) -> Tuple[np.ndarray, int]:
    """Cosmic rays: pixels sharper than the PSF/LSF in both directions (Laplacian test),
    grown along tracks. Returns (mask, number of events)."""
    from scipy import ndimage

    ny, nx = img.shape
    v = img
    lx = np.zeros_like(v)
    np.add(v[:, :-2], v[:, 2:], out=lx[:, 1:-1])
    lx[:, 1:-1] *= -0.5
    lx[:, 1:-1] += v[:, 1:-1]
    cand = np.flatnonzero((lx > 0.6 * v) & (lx > 5.0 * rn))
    mask = np.zeros((ny, nx), bool)
    if cand.size == 0:
        return mask, 0
    iy, ix = np.divmod(cand, nx)
    ok = (iy > 0) & (iy < ny - 1)
    iy, ix = iy[ok], ix[ok]

    def test(yy, xx, k_sig, k_rel):
        c = v[yy, xx]
        lxx = lx[yy, xx]
        lyy = c - 0.5 * (v[yy - 1, xx] + v[yy + 1, xx])
        sn = np.sqrt(np.maximum(c - np.maximum(lxx, lyy), 0.0) + rn**2)
        return lxx, lyy, sn, c

    lxx, lyy, sn, c = test(iy, ix, 5.0, 0.6)
    seed = (lxx > 5 * sn) & (lyy > 5 * sn) & (lxx > 0.6 * c) & (lyy > 0.6 * c)
    sy, sx = iy[seed], ix[seed]
    if sy.size == 0:
        return mask, 0
    mask[sy, sx] = True
    # grow within 2 px along tracks
    off = np.array([(a, b) for a in range(-2, 3) for b in range(-2, 3) if (a, b) != (0, 0)])
    gy = np.clip((sy[:, None] + off[None, :, 0]).ravel(), 1, ny - 2)
    gx = np.clip((sx[:, None] + off[None, :, 1]).ravel(), 0, nx - 1)
    lxx, lyy, sn, c = test(gy, gx, 3.0, 0.0)
    grow = ((lxx > 3 * sn) | (lyy > 3 * sn)) & (c > 5 * sn)
    mask[gy[grow], gx[grow]] = True
    _, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    return mask, int(n)


def _ql_run(data, header, cfg: InstrumentConfig, res: dict) -> None:
    hdr = {str(k).upper(): v for k, v in dict(header or {}).items()}
    itype = str(hdr.get("OBSTYPE", "object")).strip().lower()
    itype = _TYPE_ALIASES.get(itype, itype)
    res["image_type"] = itype
    arr = np.asarray(data)
    if arr.ndim != 2 or arr.size == 0 or min(arr.shape) < 8:
        res["flags"].append("not a 2-D image - nothing to analyse")
        return
    arr = arr.astype(float)
    ny, ntot = arr.shape
    nover = _hget(hdr, "OVSCAN", 0, int)
    if not (0 < nover < ntot // 2):
        nover = 0
    nx = ntot - nover
    raw = arr[:, :nx]
    gain = _hget(hdr, "GAIN", cfg.gain) or cfg.gain
    if nover:
        ov = arr[:, nx + 2 :] if nover > 6 else arr[:, nx:]
        yy = np.arange(ny)
        bias_y = np.polyval(np.polyfit(yy, np.median(ov, axis=1), 1), yy)
        resid = ov - bias_y[:, None]
        rn_adu = float(1.4826 * np.median(np.abs(resid - np.median(resid))))
    else:
        b = _hget(hdr, "BIASLVL", None)
        bias_y = np.full(ny, b if b is not None else float(np.percentile(raw, 0.5)))
        rn_adu = None
        res["flags"].append("no overscan - bias level taken from the header/image")
    adu = raw - bias_y[:, None]
    img = adu * gain
    rn = _hget(hdr, "RNEFF", None) or (rn_adu * gain if rn_adu else cfg.read_noise)
    fw = _hget(hdr, "FULLWELL", cfg.full_well)
    sat = (raw >= ADU_MAX - 0.5) | (img >= 0.999 * fw)
    t = _hget(hdr, "EXPTIME", None)
    res["median_adu"] = float(np.median(adu))
    res["saturated_frac"] = float(sat.mean())
    binning = (1, 1)  # total binning written by make_frame (spectral, spatial)
    try:
        b = str(hdr.get("BINNING", "")).lower().replace("x", ",").replace(" ", ",").split(",")
        b = [s for s in b if s]
        binning = (int(b[0]), int(b[1]))
    except (IndexError, ValueError):
        pass
    cfgb, bx, by = _binned(cfg, binning)
    kind = frame_kind(cfgb)
    res["frame_kind"] = kind
    ctx = dict(
        hdr=hdr,
        cfg=cfgb,
        cfg0=cfg,
        binning=binning,
        raw=raw,
        adu=adu,
        img=img,
        gain=gain,
        rn=float(rn),
        fw=fw,
        sat=sat,
        t=t,
        res=res,
        rn_adu=rn_adu,
        arr=arr,
        nx=nx,
        nover=nover,
        bias_y=bias_y,
    )
    if itype == "bias":
        _ql_bias(ctx)
        return
    if itype == "dark":
        _ql_dark(ctx)
        return
    if kind == "imaging":
        _ql_imaging(ctx)
        return
    geo = geometry(cfg, binning)
    if (geo["ny"], geo["nx"]) != (ny, nx):
        if kind == "longslit":
            geo = _geo_longslit(cfgb, bx, by, nx=nx, ny=ny)
            res["flags"].append(f"frame size differs from the standard {cfg.key} layout - using a linear model")
        else:
            res["flags"].append(f"frame layout does not match the {cfg.key} echelle model - basic statistics only")
            res["peak_adu"] = float(np.percentile(adu, 99.9))
            return
    ctx["geo"] = geo
    if itype == "flat":
        _ql_flat(ctx)
    elif itype == "arc":
        _ql_arc(ctx)
    else:
        if itype not in ("object", "sky"):
            res["flags"].append(f"unknown OBSTYPE {itype!r} - analysed as an object frame")
        _ql_object(ctx)


def _band(ctx: dict, cr: Optional[np.ndarray] = None) -> dict:
    """Frame rearranged as (rows along the slit, columns): long slit as is; echelle orders
    rectified by linear interpolation and concatenated along the column axis."""
    geo, img, sat = ctx["geo"], ctx["img"], ctx["sat"]
    ny, nx = img.shape
    if cr is None:
        cr = np.zeros_like(sat)
    if geo["kind"] == "longslit":
        yy = np.arange(ny, dtype=float)
        pos = yy[:, None] - geo["ycen"][0][None, :]
        half = geo["slit_px"] / 2.0
        slit_ok = (np.abs(yy - geo["y_mid"]) <= half - 0.8)[:, None]
        xr = int(round(geo["x_ref"]))
        return dict(
            C=img,
            pos=pos,
            slit_ok=slit_ok,
            cr=cr,
            sat=sat,
            n_o=1,
            nx=nx,
            rows_pos=yy - geo["ycen"][0][xr],
            rows_ok=slit_ok[:, 0],
            row_index=yy,
        )
    ycen = geo["ycen"]
    n_o = ycen.shape[0]
    half = geo["slit_px"] / 2.0
    J = int(np.ceil(half)) + 1
    js = np.arange(-J, J + 1)
    nj = js.size
    if "_ql_idx" not in geo:  # cache: flat pixel index / position of every band pixel
        yy = np.floor(ycen + 0.5).astype(int)[:, None, :] + js[None, :, None]  # (n_o, nj, nx)
        pos = yy - ycen[:, None, :]
        idx = np.clip(yy, 0, ny - 1) * nx + np.arange(nx)[None, None, :]
        geo["_ql_idx"] = _ro(idx.transpose(1, 0, 2).reshape(nj, n_o * nx))
        geo["_ql_pos"] = _ro(pos.transpose(1, 0, 2).reshape(nj, n_o * nx))
        geo["_ql_slit"] = _ro(np.abs(geo["_ql_pos"]) <= half - 0.8)
    idx = geo["_ql_idx"]
    return dict(
        C=img.ravel().take(idx),
        pos=geo["_ql_pos"],
        slit_ok=geo["_ql_slit"],
        cr=cr.ravel().take(idx),
        sat=sat.ravel().take(idx),
        n_o=n_o,
        nx=nx,
        rows_pos=js.astype(float),
        rows_ok=np.abs(js) <= half - 1.3,
        ycen=ycen,
    )


def _tmean(a: np.ndarray, axis: int, trim: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """Trimmed mean and sigma (from the trimmed std, Gaussian-corrected) along an axis.
    Unlike the median, not quantised when the data are integer ADU with a large gain."""
    a = np.sort(a, axis=axis)
    n = a.shape[axis]
    k = int(np.floor(trim * n))
    core = np.take(a, np.arange(k, max(n - k, k + 1)), axis=axis)
    # std of a standard normal truncated to its central (1 - 2 trim) fraction
    from scipy.stats import norm

    qn = norm.ppf(1.0 - trim) if trim > 0 else 8.0
    z = np.sqrt(max(1.0 - 2.0 * qn * norm.pdf(qn) / max(1.0 - 2.0 * trim, 1e-6), 1e-6))
    return core.mean(axis=axis), core.std(axis=axis) / z


def _binned_median(x: np.ndarray, v: np.ndarray, step: float):
    """Median of v and its error (from the MAD) in bins of x; bins with < 3 samples dropped."""
    b = np.floor((x - x.min()) / step).astype(int)
    o = np.lexsort((v, b))
    bs, vs, xs = b[o], v[o], x[o]
    starts = np.flatnonzero(np.r_[True, bs[1:] != bs[:-1]])
    cnt = np.diff(np.r_[starts, bs.size])
    grp = np.repeat(np.arange(starts.size), cnt)
    med = 0.5 * (vs[starts + (cnt - 1) // 2] + vs[starts + cnt // 2])
    xm = np.bincount(grp, weights=xs) / cnt
    dev = np.abs(vs - med[grp])
    d2 = dev[np.lexsort((dev, grp))]
    mad = 0.5 * (d2[starts + (cnt - 1) // 2] + d2[starts + cnt // 2])
    err = 1.2533 * 1.4826 * mad / np.sqrt(cnt) + 1e-9
    keep = cnt >= 3
    return xm[keep], med[keep], err[keep]


def _find_trace(Cw: np.ndarray, Pw: np.ndarray, rows_ok: np.ndarray, rn: float) -> dict:
    """Collapse a column window (median) to detect the trace, then fit a pixel-integrated
    Moffat (beta fixed) to a super-sampled profile: every pixel's position relative to the
    trace-shape line is known, so samples normalised by a smoothed per-column amplitude are
    binned in 0.25-px steps (no resampling of the data)."""
    from scipy.ndimage import median_filter, uniform_filter1d
    from scipy.optimize import curve_fit

    out = dict(found=False, signif=0.0)
    W = Cw[rows_ok]
    P = Pw[rows_ok]
    if W.shape[0] < 5 or W.shape[1] < 3:
        return out
    colsky = _tmean(W, 0, 0.25)[0]  # rows: the trace occupies a minority
    csm = float(np.median(colsky))
    good = colsky <= csm + max(0.5 * abs(csm), 5.0 * rn)
    if good.sum() < 10:
        good[:] = True
    R = W[:, good] - colsky[good]
    P = P[:, good]
    prof, sd = _tmean(R, 1, 0.2)  # columns: robust to cosmic rays / sky lines
    p_row = P.mean(axis=1)
    sig = np.maximum(1.1 * sd / np.sqrt(good.sum()), 0.05 * rn / np.sqrt(good.sum()) + 1e-9)
    nr = prof.size
    if nr >= 60:
        base = median_filter(prof, size=min(61, nr // 2 * 2 - 1), mode="mirror")
    else:
        base = np.full(nr, float(np.median(prof)))
    hp = prof - base
    h3 = uniform_filter1d(hp, 3, mode="nearest")
    s3 = h3 / (sig / np.sqrt(3.0))
    e_ = 3 if nr >= 30 else 1  # ignore the outermost rows of the band
    cand = np.nonzero(s3[e_:-e_] >= 6.0)[0] + e_
    if cand.size == 0:
        out["signif"] = float(s3[e_:-e_].max()) if nr > 2 * e_ else 0.0
        return out
    i = int(cand[np.argmax(h3[cand])])  # brightest significant row (not the most significant)
    out["signif"] = float(s3[i])
    lo = i
    while lo > 0 and hp[lo - 1] > 0.5 * hp[i]:
        lo -= 1
    hi = i
    while hi < nr - 1 and hp[hi + 1] > 0.5 * hp[i]:
        hi += 1
    fw0 = float(max(1.2, hi - lo + 1))
    near = np.abs(p_row - p_row[i]) <= max(5.0 * fw0, 8.0)
    core = np.abs(p_row - p_row[i]) <= max(2.0 * fw0, 2.0)
    amp = R[core].sum(axis=0)
    k = min(25, (amp.size // 2) * 2 - 1)
    amp_s = median_filter(amp, size=max(k, 1), mode="nearest")
    okc = amp_s > 0
    x = y = e = None
    if okc.sum() >= 5 and near.sum() >= 5:
        v = (R[near][:, okc] / amp_s[okc]).ravel()
        pp = P[near][:, okc].ravel()
        xs, ys, es = _binned_median(pp, v, 0.25)
        if xs.size >= 6:
            x, y, e = xs, ys, es
            scale = 1.0 / max(float(np.max(np.abs(y))), 1e-12)
            y, e = y * scale, e * scale
    if x is None:
        x, y, e = p_row[near], prof[near], sig[near]
        scale = 1.0

    def model(xx, A, c, fwp, b):
        al = moffat_alpha(fwp)
        return b + A * (_cdf(xx + 0.5 - c, al) - _cdf(xx - 0.5 - c, al))

    # error floor (2 % of the peak) keeps small shape systematics in the wings from dominating
    e = np.sqrt(e**2 + (0.02 * float(np.max(np.abs(y)))) ** 2)
    best = None
    for fstart in (fw0, 0.6 * fw0 + 1.0, 2.0 * fw0):
        p0 = [float(np.max(y)) * fstart * 1.1, float(p_row[i]), fstart, 0.0]
        try:
            popt, _ = curve_fit(
                model,
                x,
                y,
                p0=p0,
                sigma=e,
                maxfev=4000,
                bounds=([0.0, x.min() - 1, 0.5, -np.inf], [np.inf, x.max() + 1, float(nr), np.inf]),
            )
        except (RuntimeError, ValueError):
            continue
        chi2 = float(np.sum(((model(x, *popt) - y) / e) ** 2))
        if best is None or chi2 < best[0]:
            best = (chi2, popt)
    if best is None:
        return out
    A, c, fwp, _b = (float(q) for q in best[1])
    if not (0.7 <= fwp <= max(3.0, 0.6 * nr)) or A <= 0 or abs(c - p_row[i]) > 3 * fwp:
        return out
    out.update(found=True, center=c, fwhm_px=fwp)
    return out


def _linfit_cols(C, d, w):
    wd = w * d
    Sw = w.sum(0)
    Sd = wd.sum(0)
    Sdd = np.einsum("ij,ij->j", wd, d)
    Sy = np.einsum("ij,ij->j", w, C)
    Syd = np.einsum("ij,ij->j", wd, C)
    det = Sw * Sdd - Sd**2
    ok = (Sw >= 3) & (det > 1e-6 * np.maximum(Sw, 1) ** 2)
    b = np.where(ok, (Sw * Syd - Sd * Sy) / np.where(ok, det, 1.0), 0.0)
    a = np.where(Sw > 0, (Sy - b * Sd) / np.maximum(Sw, 1e-12), 0.0)
    return a, b


def _extract(B: dict, s0: float, fwhm_px: float, rn: float) -> dict:
    """Boxcar (the ETC's +-h FWHM aperture, fractional edges) with a clipped linear sky
    fitted across the slit on both sides of the trace."""
    C, pos, slit_ok = B["C"], B["pos"], B["slit_ok"]
    d_all = pos - s0
    h = _etc_ap_half() * fwhm_px
    inner = max(2.0 * fwhm_px, h + 1.5)
    outer = inner + max(20.0, 3.0 * fwhm_px)
    rows = np.nonzero(np.abs(d_all).min(axis=1) <= outer + 1)[0]
    if rows.size == 0:
        rows = np.arange(C.shape[0])
    r0, r1 = rows.min(), rows.max() + 1
    C, d = C[r0:r1], d_all[r0:r1]
    ok = np.broadcast_to(slit_ok, pos.shape)[r0:r1]
    crm, satm = B["cr"][r0:r1], B["sat"][r0:r1]
    sky_m = ok & (np.abs(d) >= inner) & (np.abs(d) <= outer) & ~crm & ~satm
    w = sky_m.astype(float)
    a, b = _linfit_cols(C, d, w)
    model = a + b * d
    sg2 = np.maximum(model, 0.0) + rn**2
    model -= C
    w = (sky_m & (model * model < 16.0 * sg2)).astype(float)
    a, b = _linfit_cols(C, d, w)
    sky = a + b * d
    wap = np.clip(np.minimum(d + 0.5, h) - np.maximum(d - 0.5, -h), 0.0, 1.0)
    # profile-based cosmic-ray rejection in the aperture (as in optimal extraction): each column's
    # flux from the median of its three central pixels / the Moffat profile; pixels > 5 sigma and
    # 30 % above that model are cosmic rays / hot pixels (the Laplacian test misses them on a
    # bright trace), and are replaced by the model. Columns with saturated pixels are skipped.
    al = moffat_alpha(fwhm_px)
    prof = _cdf(d + 0.5, al) - _cdf(d - 0.5, al)
    i0 = np.argmin(np.abs(d), axis=0)
    idx = np.clip(i0[None, :] + np.arange(-1, 2)[:, None], 0, C.shape[0] - 1)
    ratio = np.take_along_axis(C - sky, idx, 0) / np.maximum(np.take_along_axis(prof, idx, 0), 1e-6)
    f_col = np.median(ratio, axis=0)
    model_src = np.maximum(f_col[None, :] * prof, 0.0)
    resid = C - sky - model_src
    crp = (wap > 0) & (resid > 5.0 * np.sqrt(model_src + np.maximum(sky, 0.0) + rn**2) + 0.3 * model_src)
    crp &= ~(satm & (wap > 0)).any(axis=0)[None, :]
    fix = (wap > 0) & (crm | crp)
    crm = crm | crp
    V = C
    if fix.any():
        V = C.copy()
        V[fix] = (model_src + sky)[fix]
    S = V - sky
    flux = (wap * S).sum(0)
    var = (wap * (np.maximum(V, 0.0) + rn**2)).sum(0)
    npix = wap.sum(0)
    sky_ap = (wap * sky).sum(0)
    core = wap >= 0.5
    sat_frac = float((satm & ~crm)[core].mean()) if core.any() else 0.0
    sat_core = satm & ~crm & core
    n_sat = int(sat_core.sum())
    n_sat_cols = int(sat_core.any(axis=0).sum())
    pk = np.where(core & ~crm, V, -np.inf).max(axis=0)
    return dict(
        flux=flux, var=var, npix=npix, sky_ap=sky_ap, sat_frac=sat_frac, n_sat=n_sat, n_sat_cols=n_sat_cols, peak=pk
    )


def _bin_curve(wave, flux, var, sky_ap, dl, nbin):
    """Sum columns into nbin chunks: mean wavelength, e-/A, sky e-/A, S/N per A."""
    n = wave.size
    nb = int(max(1, min(nbin, n)))
    st = (np.arange(nb) * n) // nb
    cnt = np.diff(np.append(st, n))
    D = np.add.reduceat(dl, st)
    Fs = np.add.reduceat(flux, st)
    V = np.add.reduceat(var, st)
    W = np.add.reduceat(wave, st) / cnt
    N = np.where(V > 0, Fs / np.sqrt(np.maximum(V, 1e-30)) / np.sqrt(D), 0.0)
    return W.tolist(), (Fs / D).tolist(), (np.add.reduceat(sky_ap, st) / D).tolist(), N.tolist()


def _ql_object(ctx: dict) -> None:
    res, geo, cfg, hdr = ctx["res"], ctx["geo"], ctx["cfg"], ctx["hdr"]
    rn, gain, t = ctx["rn"], ctx["gain"], ctx["t"]
    ny, nx = ctx["img"].shape
    cr, n_cr = _cr_mask(ctx["img"], rn)
    res["cosmic_rays"] = n_cr
    if n_cr / (ny * nx) > 1.5e-3:
        res["flags"].append(f"many cosmic rays ({n_cr})")
    B = _band(ctx, cr)
    i_ref, xr = geo["i_ref"], int(round(geo["x_ref"]))
    nxo = geo["nx"]
    wave, dl, blaze = geo["wave"], geo["dlam_col"], geo["blaze"]
    D_ref = float(dl[i_ref, xr])
    pix = geo["pix"]
    # detection window around lam_ref
    halfw = max(40, nxo // 12) if geo["kind"] == "longslit" else 200
    c0, c1 = max(0, xr - halfw), min(nxo, xr + halfw + 1)
    off = i_ref * nxo
    tr = _find_trace(B["C"][:, off + c0 : off + c1], B["pos"][:, off + c0 : off + c1], B["rows_ok"], rn)
    Kc = int(
        np.clip(
            round((40.0 if geo["kind"] == "longslit" else 10.0) / D_ref), 12, 150 if geo["kind"] == "longslit" else 60
        )
    )
    win = slice(max(0, xr - Kc), min(nxo, xr + Kc + 1))
    tel = _telescope_for(cfg)
    area = tel.area_m2 * 1e4 if tel else None
    site = SITES.get(tel.site_key) if tel else None
    eta_ref = float(cfg.eff(cfg.lam_ref))
    itype = res["image_type"]
    if not tr["found"]:
        if itype != "sky":
            res["flags"].append("no trace found - target not on slit?")
        C = B["C"]
        rows = B["rows_ok"]
        skypix = _tmean(C[rows], 0, 0.2)[0].reshape(B["n_o"], nxo)
        m_, s_ = _tmean(skypix[i_ref, win], 0, 0.2)
        sky_ref = float(m_)
        sky_err = 1.1 * float(s_) / np.sqrt(skypix[i_ref, win].size)
        nb = max(1, 1000 // B["n_o"])
        Wl, Sl = [], []
        for o in range(B["n_o"]):
            W, _, S, _ = _bin_curve(wave[o], np.zeros(nxo), np.ones(nxo), skypix[o] / blaze[o].clip(0.05), dl[o], nb)
            Wl += W
            Sl += S
        res["wave"], res["sky"] = Wl, Sl
        res["units"]["sky"] = "sky e-/A per frame pixel"
        _sky_diag(ctx, sky_ref, D_ref, area, site, eta_ref, sky_err)
        return
    s0 = tr["center"]
    fwhm_px = tr["fwhm_px"]
    ex = _extract(B, s0, fwhm_px, rn)
    n_o = B["n_o"]
    flux = ex["flux"].reshape(n_o, nxo)
    var = ex["var"].reshape(n_o, nxo)
    sky_ap = ex["sky_ap"].reshape(n_o, nxo)
    npix = ex["npix"].reshape(n_o, nxo)
    # S/N in the continuum between sky lines (the ETC's sky is the continuum)
    sk = sky_ap[i_ref, win]
    clean = sk <= 1.15 * np.percentile(sk, 30) + 3.0 * rn * np.sqrt(np.maximum(npix[i_ref, win], 1.0))
    if clean.sum() < 8:
        clean = np.ones_like(sk, bool)
    Sm = float(np.median(flux[i_ref, win][clean]))
    Vm = float(np.median(var[i_ref, win][clean]))
    snr_col = Sm / np.sqrt(Vm) if (Vm > 0 and Sm > 0) else 0.0
    snr_A = snr_col / np.sqrt(D_ref)
    dn = geo["dlam_native"]
    if geo["kind"] == "longslit":
        trace_row = float(geo["ycen"][0][xr] + s0)
    else:
        trace_row = float(geo["ycen"][i_ref, xr] + s0)
    spc = (sky_ap[i_ref, win] / np.maximum(npix[i_ref, win], 1e-6))[clean]
    m_, s_ = _tmean(spc, 0, 0.2)
    sky_pix, sky_err = float(m_), 1.1 * float(s_) / np.sqrt(spc.size)
    res.update(
        trace_found=True,
        trace_pos=trace_row,
        fwhm_arcsec=float(fwhm_px * pix),
        peak_adu=_trace_peak(ex["peak"], n_o, nxo) / gain,
        peak_adu_raw=float(np.max(ex["peak"]) / gain),
        saturated_frac=ex["sat_frac"],
        sky_adu_per_pix=sky_pix / gain,
        counts_ref_e=Sm * dn / D_ref,
        snr_ref_per_A=float(snr_A),
        snr_ref_per_pix=float(snr_A * np.sqrt(dn)),
    )
    # curves
    if geo["kind"] == "longslit":
        W, F, S, N = _bin_curve(wave[0], flux[0], var[0], sky_ap[0], dl[0], min(1000, nxo // 2))
    else:
        W, F, S, N = [], [], [], []
        G, orders = geo["G"], geo["orders"]
        nb = max(1, 1000 // n_o)
        for o in range(n_o):
            m = orders[o]
            sel = (wave[o] >= G / (m + 0.5)) & (wave[o] < G / (m - 0.5)) & (blaze[o] > 0.3)
            if sel.sum() < 4:
                continue
            bz = blaze[o, sel]
            w_, f_, s_, n_ = _bin_curve(
                wave[o, sel], flux[o, sel] / bz, var[o, sel] / bz**2, sky_ap[o, sel] / bz, dl[o, sel], nb
            )
            W += w_
            F += f_
            S += s_
            N += n_
        o = i_ref
        w_, f_, s_, n_ = _bin_curve(wave[o], flux[o], var[o], sky_ap[o], dl[o], nxo // 2)
        res["order_ref"] = {
            "m": int(orders[o]),
            "wave": w_,
            "flux": f_,
            "sky": s_,
            "snr_curve": n_,
            "note": "raw (not blaze-corrected) extraction of the order containing lam_ref",
        }
    res["wave"], res["flux"], res["sky"], res["snr_curve"] = W, F, S, N
    # flags
    if ex["n_sat_cols"] >= 2:
        res["flags"].append(
            f"saturated pixels in trace ({ex['n_sat']} px, {100 * ex['sat_frac']:.2f}% of the trace core)"
        )
    elif _trace_peak(ex["peak"], n_o, nxo) > LINEARITY * ctx["fw"]:
        res["flags"].append("trace peak above the linearity limit")
    exp = _hget(hdr, "EXPSN", None)
    if exp is not None and exp > 0 and snr_A < 0.6 * exp:
        res["flags"].append(f"trace much fainter than expected (S/N/A {snr_A:.1f} vs predicted {exp:.1f})")
    if fwhm_px * pix > 2.0 and ex["n_sat"] == 0:
        res["flags"].append(f'broad trace (FWHM {fwhm_px * pix:.1f}") - seeing, focus or guiding?')
    if geo["kind"] == "echelle" or geo["slit_px"] < geo.get("ny", 1e9):
        if abs(s0) > 0.3 * geo["slit_px"]:
            res["flags"].append("trace near the end of the slit")
    # the target's Moffat wings in the sky region (short slits, bright targets) bias the sky high
    al = moffat_alpha(fwhm_px)
    d_sky = max(2.0 * fwhm_px, _etc_ap_half() * fwhm_px + 1.5) + 1.0
    wing = (
        Sm
        / max(float(_cdf(_etc_ap_half() * fwhm_px, al) * 2.0), 1e-6)
        * float(_cdf(d_sky + 0.5, al) - _cdf(d_sky - 0.5, al))
    )
    if wing < 0.3 * max(sky_pix - rn, 1e-9):
        _sky_diag(ctx, sky_pix, D_ref, area, site, eta_ref, sky_err)
    else:
        res["sky_adu_per_pix"] = sky_pix / gain
    if t and t > 0 and area and site is not None and Sm > 0:
        X = _hget(hdr, "AIRMASS", 1.2) or 1.2
        trans = float(10.0 ** (-0.4 * float(extinction_coeff(site, cfg.lam_ref)) * X))
        fw_as = fwhm_px * pix
        hap = _etc_ap_half() * fw_as
        fap = float(strip_fraction(-hap, hap, fw_as))
        fsl = float(slit_fraction(cfg.slit_width, fw_as, 0.0))
        n = Sm / (t * D_ref * area * eta_ref * trans * fap * fsl)
        res["mag_ref_est"] = float(-2.5 * np.log10(n * H_CGS * cfg.lam_ref) - 48.6)


def _pedestal_ref(ctx) -> float:
    """Echelle: scattered-light pedestal (e-) near lam_ref from the inter-order gaps."""
    geo = ctx["geo"]
    if geo["kind"] != "echelle":
        return 0.0
    ycen, i = geo["ycen"], geo["i_ref"]
    xr = int(round(geo["x_ref"]))
    cols = np.arange(max(0, xr - 100), min(geo["nx"], xr + 100))
    gaps = []
    for j in (i - 1, i):
        if 0 <= j < ycen.shape[0] - 1:
            gaps.append(np.rint(0.5 * (ycen[j, cols] + ycen[j + 1, cols])).astype(int))
    if not gaps:
        return 0.0
    vals = np.concatenate([ctx["img"][g, cols] for g in gaps])
    return float(max(_tmean(vals, 0, 0.2)[0], 0.0))


def _trace_peak(pk: np.ndarray, n_o: int, nx: int) -> float:
    """Brightest level along the trace (e- per frame pixel): per-column core maxima smoothed by a
    5-column running median along the dispersion (per order), so an isolated cosmic ray or hot
    pixel sitting on the trace does not set the peak."""
    from scipy.ndimage import median_filter

    a = np.where(np.isfinite(pk), pk, 0.0).reshape(n_o, nx)
    return float(median_filter(a, size=(1, 5), mode="nearest").max())


def _sky_diag(ctx, sky_pix_e, D_ref, area, site, eta_ref, err_e: float = 0.0):
    """Sky surface brightness at lam_ref from the frame; flag bright (Moon/twilight) skies
    only when the sky is detected (> 3 sigma) and the 2-sigma lower bound is > 1 mag above dark."""
    res, cfg, geo, t = ctx["res"], ctx["cfg"], ctx["geo"], ctx["t"]
    res["sky_adu_per_pix"] = float(sky_pix_e / ctx["gain"])
    if not (t and t > 0 and area and site is not None and eta_ref > 0):
        return
    if geo["kind"] == "echelle":  # inter-order level = dark + scattered light
        base = _pedestal_ref(ctx)
    else:
        base = cfg.dark_e_per_hr / 3600.0 * cfg.bin_spatial * cfg.bin_spectral * geo["r"] * geo["q"] * t
    conv = t * D_ref * cfg.slit_width * geo["pix"] * area * eta_ref
    net = sky_pix_e - base
    if net <= 3.0 * err_e or net <= 0:
        return

    def mag(e):
        m_ = float(-2.5 * np.log10(e / conv * H_CGS * cfg.lam_ref) - 48.6)
        if cfg.nir and cfg.sky_line_factor > 0:
            m_ += 2.5 * np.log10(cfg.sky_line_factor)
        return m_

    m = mag(net)
    res["sky_mag_ref"] = m
    dark_mag = float(dark_sky_zenith(site, cfg.lam_ref))
    if mag(net - 2.0 * err_e) < dark_mag - 1.0:
        res["flags"].append(f"high sky - Moon or twilight ({m:.1f} vs dark {dark_mag:.1f} AB mag/arcsec^2)")


def _illuminated(ctx) -> Tuple[np.ndarray, Optional[dict]]:
    """Values (ADU, bias-subtracted) of well-illuminated slit pixels, and the band (echelle)."""
    geo, cfg = ctx["geo"], ctx["cfg"]
    if geo["kind"] == "longslit":
        yy = np.arange(ctx["img"].shape[0])
        rows = np.abs(yy - geo["y_mid"]) <= geo["slit_px"] / 2.0 - 1.5
        eff = cfg.eff(geo["wave"][0])
        cols = eff > 0.25 * eff.max()
        return ctx["adu"][np.ix_(rows, cols)], None
    B = _band(ctx)
    js = B["rows_pos"]
    rows = np.abs(js) <= geo["slit_px"] / 2.0 - 1.5
    cols = (geo["blaze"] > 0.5).ravel()
    return B["C"][np.ix_(rows, cols)] / ctx["gain"], B


def _ql_flat(ctx) -> None:
    res, geo = ctx["res"], ctx["geo"]
    vals, B = _illuminated(ctx)
    if vals.size == 0:
        res["flags"].append("no illuminated region found")
        return
    peak = float(np.percentile(vals, 99.5))
    med = float(np.median(vals))
    res.update(peak_adu=peak, median_adu=med)
    res["saturated_frac"] = float(np.mean(vals >= min(ADU_MAX - BIAS_ADU, ctx["fw"] / ctx["gain"]) * 0.999))
    lim = min(ADU_MAX - BIAS_ADU - 500.0, LINEARITY * ctx["fw"] / ctx["gain"])
    if peak < 200:
        res["flags"].append("no flat illumination - lamp off or shutter closed?")
    elif peak < 5000:
        res["flags"].append(f"flat too faint (peak {peak:.0f} ADU)")
    if res["saturated_frac"] > 0:
        res["flags"].append("flat saturated")
    elif peak > lim:
        res["flags"].append(f"flat above the linearity limit (peak {peak:.0f} ADU)")
    if B is not None:
        n_o, nx = geo["wave"].shape
        js = B["rows_pos"]
        rows = np.abs(js) <= geo["slit_px"] / 2.0 - 1.5
        C = B["C"][rows].reshape(rows.sum(), n_o, nx) / ctx["gain"]
        om = np.array([np.median(C[:, o, geo["blaze"][o] > 0.5]) for o in range(n_o)])
        res["order_levels_adu"] = [float(v) for v in om]
        n_low = int((om < 1000).sum())
        if n_low > max(2, n_o // 4) and peak >= 5000:
            res["flags"].append(f"{n_low} orders below 1000 ADU (lamp weak at the ends of the range)")


def _peaks(spec: np.ndarray, rn: float, nrows: int) -> np.ndarray:
    from scipy.ndimage import median_filter

    bkg = median_filter(spec, size=41, mode="nearest")
    r = spec - bkg
    sig = np.sqrt(np.maximum(bkg, 0.0) / max(nrows, 1) + rn**2 / max(nrows, 1)) + 1.4826 * np.median(np.abs(r)) * 0.2
    pk = np.zeros(spec.size, bool)
    pk[1:-1] = (r[1:-1] > 10 * sig[1:-1]) & (r[1:-1] >= r[:-2]) & (r[1:-1] > r[2:])
    return np.nonzero(pk)[0]


def _ql_arc(ctx) -> None:
    res, geo, rn = ctx["res"], ctx["geo"], ctx["rn"]
    img = ctx["img"]
    if geo["kind"] == "longslit":
        yy = np.arange(img.shape[0])
        rows = np.abs(yy - geo["y_mid"]) <= min(10.0, geo["slit_px"] / 4.0)
        specs = np.median(img[rows], axis=0)[None, :]
        nrows = int(rows.sum())
        region = np.abs(yy - geo["y_mid"]) <= geo["slit_px"] / 2.0 - 1.0
        sat_n = int(ctx["sat"][region].sum())
        peak = float(ctx["adu"][region].max())
    else:
        B = _band(ctx)
        js = B["rows_pos"]
        rows = np.abs(js) <= 2
        n_o, nx = geo["wave"].shape
        specs = B["C"][rows].mean(axis=0).reshape(n_o, nx)
        nrows = int(rows.sum())
        inside = np.abs(js) <= geo["slit_px"] / 2.0 - 1.0
        sat_n = int(B["sat"][inside].sum())
        peak = float(B["C"][inside].max() / ctx["gain"])
    counts = [len(_peaks(s, rn, nrows)) for s in specs]
    n_lines = int(sum(counts))
    res.update(peak_adu=peak, n_lines=n_lines, n_lines_ref=int(counts[geo["i_ref"]]), saturated_pixels=sat_n)
    if geo["kind"] == "longslit":
        res["line_wave"] = [float(geo["wave"][0][i]) for i in _peaks(specs[0], rn, nrows)][:200]
    if sat_n > 0:
        res["flags"].append(f"arc lines saturated ({sat_n} pixels)")
    if n_lines < 3:
        res["flags"].append("no arc lines - lamp off?")
    elif peak < 2000:
        res["flags"].append(f"arc too faint (brightest line {peak:.0f} ADU)")


def _ql_bias(ctx) -> None:
    res, hdr = ctx["res"], ctx["hdr"]
    raw, arr, nx = ctx["raw"], ctx["arr"], ctx["nx"]
    level = float(np.median(raw))
    res["median_adu"] = level
    res["peak_adu"] = float(np.percentile(ctx["adu"], 99.99))
    if ctx["nover"]:
        ovl = float(np.median(arr[:, nx:]))
        res["overscan_adu"] = ovl
        if abs(level - ovl) > 5.0:
            res["flags"].append(
                f"image level differs from the overscan by {level - ovl:.1f} ADU - light leak or bias drift?"
            )
    rn_img = float(1.4826 * np.median(np.abs(ctx["adu"] - np.median(ctx["adu"]))))
    res["read_noise_adu"] = rn_img
    exp = (_hget(hdr, "RNEFF", None) or ctx["cfg"].read_noise) / ctx["gain"]
    if rn_img > 1.5 * exp:
        res["flags"].append(f"read noise high ({rn_img:.1f} ADU vs {exp:.1f} expected)")
    if not (100.0 <= level <= 10000.0):
        res["flags"].append(f"bias level unusual ({level:.0f} ADU)")


def _ql_dark(ctx) -> None:
    res, cfg, t = ctx["res"], ctx["cfg"], ctx["t"]
    img = ctx["img"]
    med0 = float(np.median(img))
    sg0 = 1.4826 * float(np.median(np.abs(img - med0))) + 1e-6
    med = float(np.mean(img[np.abs(img - med0) < 5 * sg0]))  # clipped mean: unbiased by ADU quantisation
    res["median_adu"] = med / ctx["gain"]
    res["peak_adu"] = float(np.percentile(ctx["adu"], 99.99))
    sig = np.sqrt(max(med, 0.0) + ctx["rn"] ** 2)
    res["hot_pixels"] = int((img > med + 10 * sig).sum())
    cr, n_cr = _cr_mask(img, ctx["rn"])
    res["cosmic_rays"] = n_cr
    if t and t > 0:
        rate = med / t * 3600.0
        res["dark_e_per_hr"] = rate
        try:
            geo = geometry(ctx["cfg0"], ctx["binning"]) if frame_kind(cfg) != "imaging" else {"r": 1.0, "q": 1.0}
            rq = geo["r"] * geo["q"]
        except Exception:
            rq = 1.0
        exp = cfg.dark_e_per_hr * cfg.bin_spatial * cfg.bin_spectral * rq
        if rate > 3.0 * exp + 5.0:
            res["flags"].append(f"high dark current ({rate:.0f} e-/pix/hr vs {exp:.0f}) - light leak?")


def _ql_imaging(ctx) -> None:
    from scipy.ndimage import gaussian_filter
    from scipy.optimize import curve_fit

    res, cfg, rn, gain = ctx["res"], ctx["cfg"], ctx["rn"], ctx["gain"]
    img = ctx["img"]
    ny, nx = img.shape
    pix = cfg.spatial_scale
    itype = res["image_type"]
    sky = float(np.median(img))
    sig = float(1.4826 * np.median(np.abs(img - sky)))
    res["sky_adu_per_pix"] = sky / gain
    res["peak_adu"] = float(np.percentile(ctx["adu"], 99.99))
    if itype == "flat":
        pk = float(np.percentile(ctx["adu"], 99.5))
        res["peak_adu"] = pk
        if pk < 5000:
            res["flags"].append(f"flat too faint (peak {pk:.0f} ADU)")
        if ctx["sat"].any():
            res["flags"].append("flat saturated")
        return
    cr, n_cr = _cr_mask(img, rn)
    res["cosmic_rays"] = n_cr
    work = np.where(cr, sky, img) - sky
    sm = gaussian_filter(work, 1.5)
    y0, y1, x0, x1 = ny // 4, 3 * ny // 4, nx // 4, 3 * nx // 4
    sub = sm[y0:y1, x0:x1]
    k = int(np.argmax(sub))
    yc, xc = y0 + k // sub.shape[1], x0 + k % sub.shape[1]
    noise = max(sig, 1e-6) / (2.0 * np.sqrt(np.pi) * 1.5)
    if sm[yc, xc] / noise < 10:
        if itype == "object":
            res["flags"].append("no source found near the field centre")
        return
    K = 15
    ys, xs = slice(max(0, yc - K), min(ny, yc + K + 1)), slice(max(0, xc - K), min(nx, xc + K + 1))
    st = work[ys, xs]
    yy, xx = np.mgrid[ys, xs]
    r = np.hypot(yy - yc, xx - xc).ravel()

    def model(rr, A, al):
        return A * (1.0 + (rr / al) ** 2) ** -BETA

    try:
        (A, al), _ = curve_fit(model, r, st.ravel(), p0=[float(st.max()), 2.0], maxfev=2000)
        fwp = float(2.0 * abs(al) * np.sqrt(2.0 ** (1.0 / BETA) - 1.0))
    except (RuntimeError, ValueError):
        fwp = 3.0
    r_ap = 0.8 * fwp
    ap = r <= r_ap
    flux = float(st.ravel()[ap].sum())
    var = float((np.maximum(st.ravel()[ap] + sky, 0) + rn**2).sum())
    res.update(
        trace_found=True,
        trace_pos=float(yc),
        fwhm_arcsec=fwp * pix,
        counts_ref_e=flux,
        snr_ref_per_pix=flux / np.sqrt(var) if var > 0 else 0.0,
        snr_ref_per_A=None,
        peak_adu=float((img[ys, xs].max()) / gain),
        saturated_frac=float(ctx["sat"][ys, xs].mean()),
    )
    res["units"]["snr_ref_per_pix"] = "aperture S/N (r = 0.8 FWHM)"
    if ctx["sat"][ys, xs].any():
        res["flags"].append("saturated pixels in the target")


# ---------------------------------------------------------------------------
# FITS I/O
# ---------------------------------------------------------------------------
_STRUCTURAL = {"SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "EXTEND", "BZERO", "BSCALE", "PCOUNT", "GCOUNT"}


def write_fits(path, data: np.ndarray, header: dict) -> None:
    from astropy.io import fits

    hdu = fits.PrimaryHDU(np.asarray(data))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, v in dict(header or {}).items():
            key = str(k).upper()
            if key in _STRUCTURAL or v is None:
                continue
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = float(v)
            elif isinstance(v, (np.bool_,)):
                v = bool(v)
            elif not isinstance(v, (int, float, str, bool)):
                v = str(v)
            if isinstance(v, float) and not np.isfinite(v):
                v = str(v)
            hdu.header[key] = v
        hdu.writeto(path, overwrite=True)


def read_fits(path) -> Tuple[np.ndarray, dict]:
    from astropy.io import fits

    with fits.open(path) as hl:
        data = np.array(hl[0].data)
        hdr = {k: v for k, v in hl[0].header.items() if k not in ("COMMENT", "HISTORY", "")}
    return data, hdr
