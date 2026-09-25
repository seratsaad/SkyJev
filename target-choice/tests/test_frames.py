"""Tests for obsassist.sim.frames: synthetic raw frames, FITS I/O and the quick-look."""

import time

import numpy as np
import pytest

from obsassist.astro.sites import CLAY, KECK1, KECK2, LAS_CAMPANAS, MAUNAKEA
from obsassist.astro.sky import delivered_fwhm
from obsassist.etc import Source, compute_rates
from obsassist.instruments.base import get_config
from obsassist.sim.frames import (
    TEMPLATES,
    FrameRequest,
    geometry,
    make_frame,
    quicklook,
    read_fits,
    typical_cal_exptime,
    write_fits,
)

LRIS = ("LRIS-R400", KECK1, MAUNAKEA)
MIKE = ("MIKE-RED", CLAY, LAS_CAMPANAS)


def _req(which, image_type="object", t=600.0, **kw):
    key, tel, site = which
    return FrameRequest(get_config(key), image_type, t, site, tel, **kw)


def _frame(which, image_type="object", t=600.0, **kw):
    data, hdr, truth = make_frame(_req(which, image_type, t, **kw))
    return data, hdr, truth, quicklook(data, hdr, get_config(which[0]))


# ---------------------------------------------------------------- shapes, headers, geometry
@pytest.mark.parametrize(
    "which",
    [
        LRIS,
        MIKE,
        ("LRIS-B600", KECK1, MAUNAKEA),
        ("MIKE-BLUE", CLAY, LAS_CAMPANAS),
        ("HIRES-C2", KECK1, MAUNAKEA),
        ("MOSFIRE-H", KECK1, MAUNAKEA),
        ("DEIMOS-600ZD", KECK2, MAUNAKEA),
        ("LRIS-IMG-R", KECK1, MAUNAKEA),
    ],
)
@pytest.mark.parametrize("itype", ["object", "flat", "arc", "bias", "dark", "sky"])
def test_shapes_dtypes_headers(which, itype):
    data, hdr, truth = make_frame(_req(which, itype, 30.0, source=Source(18.0), seed=1))
    assert data.dtype == np.uint16 and data.ndim == 2
    assert data.shape == (hdr["NAXIS2"], hdr["NAXIS1"])
    assert all(len(k) <= 8 for k in hdr)
    assert hdr["OBSTYPE"] == itype.upper()
    over = data[:, -hdr["OVSCAN"] :].astype(float)
    assert 990 < np.median(over) < 1010  # bias + read noise only
    assert over.std() < 10
    assert truth["image_type"] == itype


def test_geometry_echelle_mike_red():
    cfg = get_config("MIKE-RED")
    g = geometry(cfg)
    n_o = len(g["orders"])
    assert 30 <= n_o <= 36 and g["kind"] == "echelle"
    assert g["wave"][g["i_ref"], int(round(g["x_ref"]))] == pytest.approx(cfg.lam_ref, rel=1e-4)
    assert np.all(np.diff(g["wave"], axis=1) > 0)
    gaps = np.diff(g["ycen"], axis=0)
    assert gaps.min() > g["slit_px"]  # orders do not overlap
    mid = gaps[:, g["nx"] // 2]
    assert mid[-1] > mid[0]  # spacing grows with order number
    sag = g["ycen"][:, [0, -1]].mean(axis=1) - g["ycen"][:, g["nx"] // 2]
    assert np.all(sag > 5)  # curved orders
    assert g["blaze"][g["i_ref"]].max() == pytest.approx(1.0, abs=1e-3)


def test_geometry_longslit_lris():
    cfg = get_config("LRIS-R400")
    g = geometry(cfg)
    assert g["nx"] == 2048 and g["ny"] <= 400
    w = g["wave"][0]
    assert w[0] == pytest.approx(cfg.lam_min) and w[-1] == pytest.approx(cfg.lam_max)
    assert np.all(np.diff(w) > 0)
    assert g["r"] == pytest.approx(g["dlam_col"][0, int(round(g["x_ref"]))] / cfg.dlam_pix, rel=1e-3)


def test_deterministic_seed():
    a = make_frame(_req(LRIS, source=Source(19.0), seed=7))[0]
    b = make_frame(_req(LRIS, source=Source(19.0), seed=7))[0]
    c = make_frame(_req(LRIS, source=Source(19.0), seed=8))[0]
    assert np.array_equal(a, b) and not np.array_equal(a, c)


# ---------------------------------------------------------------- object frames
@pytest.mark.parametrize("which", [LRIS, MIKE])
def test_no_trace_when_off_target(which):
    for seed in range(3):
        _, _, truth, q = _frame(which, source=Source(18.0), on_target=False, seed=seed)
        assert not q["trace_found"]
        assert any("no trace" in f for f in q["flags"])
        assert q["snr_ref_per_A"] == 0.0
        assert truth["trace_row"] is None


@pytest.mark.parametrize("which", [LRIS, MIKE])
def test_missed_slit_offset(which):
    _, _, _, q = _frame(which, source=Source(18.0), slit_offset_arcsec=3.0, seed=2)
    assert not q["trace_found"]


def test_saturation_flag_bright_star():
    _, _, _, q = _frame(LRIS, t=600.0, source=Source(9.0), seed=1)
    assert q["trace_found"]
    assert q["saturated_frac"] > 0.05
    assert any("saturated" in f for f in q["flags"])
    _, _, _, q2 = _frame(MIKE, t=1800.0, source=Source(8.0), seed=1)
    assert any("saturated" in f for f in q2["flags"])


@pytest.mark.parametrize(
    "which,mag,sky", [(LRIS, 22.0, 21.0), (LRIS, 17.0, 21.0), (MIKE, 18.5, 20.8), (MIKE, 13.0, 20.8)]
)
def test_snr_consistency_with_etc(which, mag, sky):
    key, tel, site = which
    cfg = get_config(key)
    src = Source(mag)
    rates = compute_rates(cfg, tel, site, src, seeing_500=0.7, airmass=1.2, sky_mag=sky, unit="per_A")
    etc_snr = float(rates.snr(600.0))
    fwhm = float(rates.fwhm)
    for seed in (0, 1):
        data, hdr, truth, q = _frame(which, source=src, fwhm_arcsec=fwhm, airmass=1.2, sky_mag_ref=sky, seed=seed)
        assert truth["snr_ref_per_A"] == pytest.approx(etc_snr, rel=1e-6)  # same physics
        assert q["trace_found"]
        assert q["snr_ref_per_A"] == pytest.approx(etc_snr, rel=0.20)
        assert q["snr_ref_per_pix"] == pytest.approx(q["snr_ref_per_A"] * np.sqrt(cfg.dlam_pix), rel=1e-6)
        assert q["fwhm_arcsec"] == pytest.approx(fwhm, rel=0.10)
        assert q["trace_pos"] == pytest.approx(truth["trace_row"], abs=0.5)
        assert q["counts_ref_e"] == pytest.approx(truth["signal_e_ap_native"], rel=0.2)
        assert q["mag_ref_est"] == pytest.approx(mag, abs=0.15)
        assert len(q["wave"]) == len(q["flux"]) == len(q["sky"]) == len(q["snr_curve"]) <= 1000
        assert not any("saturated" in f for f in q["flags"])


def test_expected_snr_flag():
    data, hdr, _, q = _frame(LRIS, source=Source(21.0), seed=3)
    hdr = dict(hdr, EXPSN=3.0 * q["snr_ref_per_A"])
    q2 = quicklook(data, hdr, get_config("LRIS-R400"))
    assert any("fainter than expected" in f for f in q2["flags"])
    hdr = dict(hdr, EXPSN=1.0 * q["snr_ref_per_A"])
    assert not any("fainter than expected" in f for f in quicklook(data, hdr, get_config("LRIS-R400"))["flags"])


def test_high_sky_flag_and_sky_level():
    _, _, _, q = _frame(MIKE, source=Source(17.0), sky_mag_ref=18.0, seed=4)
    assert any("high sky" in f for f in q["flags"])
    assert q["sky_mag_ref"] == pytest.approx(18.0, abs=0.3)
    _, _, _, q = _frame(LRIS, source=Source(20.0), sky_mag_ref=21.0, seed=4)
    assert not any("high sky" in f for f in q["flags"])
    assert q["sky_mag_ref"] == pytest.approx(21.0, abs=0.3)


@pytest.mark.parametrize("template", TEMPLATES)
def test_templates_render(template):
    data, hdr, truth, q = _frame(LRIS, source=Source(18.0), template=template, z=0.05, seed=5)
    assert np.isfinite(np.asarray(q["flux"], dtype=float)).all()
    assert q["trace_found"]


def test_qso_lyman_forest():
    _, _, _, q = _frame(LRIS, t=1200.0, source=Source(19.0), template="qso", z=6.0, seed=6)
    w, f = np.array(q["wave"]), np.array(q["flux"])
    lya = 1215.67 * 7.0
    blue = np.median(f[(w > 7000) & (w < lya - 150)])
    red = np.median(f[(w > lya + 150) & (w < 9300)])
    assert blue < 0.2 * red


def test_cosmic_rays_counted():
    _, _, truth, q = _frame(LRIS, t=1800.0, source=Source(20.0), seed=9)
    assert truth["n_cosmic"] > 100
    assert 0.5 * truth["n_cosmic"] < q["cosmic_rays"] < 2.5 * truth["n_cosmic"]


# ---------------------------------------------------------------- peak pixel vs the ETC (planner's saturation model)
def _etc_peak_max(cfg, tel, site, src, seeing, X, t, sky, lo=None, hi=None):
    lo = cfg.lam_min if lo is None else lo
    hi = cfg.lam_max if hi is None else hi
    lams = [float(v) for v in np.linspace(lo, hi, 41)[1:-1] if cfg.eff(v) > 0]
    return max(
        float(compute_rates(cfg, tel, site, src, seeing_500=seeing, airmass=X, lam=v, sky_mag=sky).peak_pix) * t
        for v in lams
    )


def _ql_peak_native(which, src, template, seeing, X, t, sky, seed=1):
    key, tel, site = which
    cfg = get_config(key)
    fw = float(delivered_fwhm(seeing, X, cfg.lam_ref, tel.iq_floor_arcsec, 0.0, tel.outer_scale_m))
    req = _req(
        which,
        "object",
        t,
        source=src,
        template=template,
        fwhm_arcsec=fw,
        airmass=X,
        sky_mag_ref=sky,
        seed=seed,
        binning=(cfg.bin_spatial, cfg.bin_spectral),
    )  # as the console passes it
    data, hdr, _ = make_frame(req)
    q = quicklook(data, hdr, cfg)
    return q["peak_adu"] * hdr["GAINNAT"], q  # native-pixel e-


def test_peak_pixel_lead_case_mike_red_cool_star():
    """MIKE-RED, V=13.2 (Vega) 4700 K star, 'galaxy' template, 0.6" seeing, X=1.1, 1800 s: the
    frame's trace peak matches the ETC's brightest pixel over 5200-9000 A and does not saturate
    (the template adds structure only, the continuum slope is the SED's; binning is counted once)."""
    cfg = get_config("MIKE-RED")
    src = Source(13.2, band="V", system="Vega", sed="bb:4700")
    etc = _etc_peak_max(cfg, CLAY, LAS_CAMPANAS, src, 0.6, 1.1, 1800.0, 21.0, 5200.0, 9000.0)
    ql, q = _ql_peak_native(MIKE, src, "galaxy", 0.6, 1.1, 1800.0, 21.0)
    # measured -6%: pixel integration of the line-spread profile lowers the frame's peak
    assert ql == pytest.approx(etc, rel=0.10)
    assert not any("saturated" in f for f in q["flags"])


@pytest.mark.parametrize(
    "which",
    [
        MIKE,
        ("MIKE-BLUE", CLAY, LAS_CAMPANAS),
        ("HIRES-C2", KECK1, MAUNAKEA),
        LRIS,
        ("LDSS3-VPHALL", CLAY, LAS_CAMPANAS),
    ],
)
@pytest.mark.parametrize("sed,template", [("bb:4700", "galaxy"), ("bb:14000", "hot_star"), ("flat_fnu", "flat")])
def test_peak_pixel_matches_etc(which, sed, template):
    key, tel, site = which
    cfg = get_config(key)
    t, seeing, X, sky = 1800.0, 0.6, 1.1, 21.0
    p0 = _etc_peak_max(cfg, tel, site, Source(13.0, band="V", system="Vega", sed=sed), seeing, X, t, sky)
    mag = 13.0 + 2.5 * np.log10(p0 / (0.4 * cfg.full_well))  # ETC peak at 40% of full well
    src = Source(float(mag), band="V", system="Vega", sed=sed)
    etc = _etc_peak_max(cfg, tel, site, src, seeing, X, t, sky)
    ql, q = _ql_peak_native(which, src, template, seeing, X, t, sky)
    # measured 0.86 .. 1.07: HIRES frame rows average 2 native rows (lower peak)
    assert 0.83 < ql / etc < 1.10
    assert not any("saturated" in f for f in q["flags"])


def test_binning_semantics():
    """FrameRequest.binning is the total on-chip binning; the configuration's own binning in
    either axis order (as the console passes it) or (1, 1) means no extra binning."""
    for key, tel, site in [("HIRES-C2", KECK1, MAUNAKEA), ("MIKE-RED", CLAY, LAS_CAMPANAS), LRIS]:
        cfg = get_config(key)
        ref = make_frame(_req((key, tel, site), "bias", 0.0, seed=2))[1]
        for b in [(1, 1), (cfg.bin_spectral, cfg.bin_spatial), (cfg.bin_spatial, cfg.bin_spectral)]:
            h = make_frame(_req((key, tel, site), "bias", 0.0, seed=2, binning=b))[1]
            assert (h["NAXIS1"], h["NAXIS2"], h["BINNING"]) == (ref["NAXIS1"], ref["NAXIS2"], ref["BINNING"])
    cfg = get_config("LRIS-R400")
    h = make_frame(_req(LRIS, "bias", 0.0, binning=(2, 2)))[1]
    assert h["BINNING"] == "2,2" and h["NAXIS1"] == 1024 + 32


def test_template_does_not_change_continuum_slope():
    """Templates add features only; the continuum comes from Source.sed (as in the ETC)."""
    from obsassist.sim.frames import template_shape

    lam = np.array([5200.0, 6500.0, 8900.0])
    for name in TEMPLATES:
        shp = template_shape(name, lam, 0.0, 3.0, 6500.0)
        if name in ("flat", "galaxy", "hot_star", "sn_ii", "white_dwarf"):  # (m_dwarf: TiO bands there)
            assert np.all(np.abs(shp[[0, 2]] - 1.0) < 0.15), name


# ---------------------------------------------------------------- calibrations
@pytest.mark.parametrize("which", [LRIS, MIKE, ("LRIS-B600", KECK1, MAUNAKEA), ("MOSFIRE-K", KECK1, MAUNAKEA)])
def test_flat_levels(which):
    cfg = get_config(which[0])
    t0 = typical_cal_exptime(cfg, "flat")
    _, _, _, q = _frame(which, "flat", t0, seed=1)
    lim = min(40000.0, 0.8 * cfg.full_well / cfg.gain)
    assert 0.5 * min(30000.0, 0.6 * cfg.full_well / cfg.gain) < q["peak_adu"] < lim
    assert not any(("too faint" in f) or ("saturated" in f) for f in q["flags"])
    _, _, _, qf = _frame(which, "flat", 0.03 * t0, seed=1)
    assert any("flat too faint" in f for f in qf["flags"])


def test_arc_levels_and_lines():
    cfg = get_config("MIKE-RED")
    _, _, _, q = _frame(MIKE, "arc", typical_cal_exptime(cfg, "arc"), seed=1)
    assert q["n_lines_ref"] >= 15 and q["n_lines"] > 300  # ThAr: many lines in every order
    assert 10000 < q["peak_adu"] < 60000
    assert not any("saturated" in f for f in q["flags"])
    cfg = get_config("LRIS-R400")
    _, _, _, q = _frame(LRIS, "arc", typical_cal_exptime(cfg, "arc"), seed=1)
    assert q["n_lines"] >= 30
    lw = np.array(q["line_wave"])
    assert np.min(np.abs(lw - 6965.4)) < 5.0  # Ar I 6965 identified near its place
    _, _, _, q = _frame(LRIS, "arc", 10 * typical_cal_exptime(cfg, "arc"), seed=1)
    assert any("arc lines saturated" in f for f in q["flags"])


def test_bias_and_dark():
    _, _, _, q = _frame(MIKE, "bias", 0.0, seed=1)
    assert 995 < q["median_adu"] < 1012 and not q["flags"]
    cfg = get_config("MIKE-RED")
    _, hdr, _, q = _frame(MIKE, "dark", 3600.0, seed=1)
    exp = cfg.dark_e_per_hr * cfg.bin_spatial * cfg.bin_spectral * hdr["NATPIXX"] * hdr["NATPIXY"]
    assert q["dark_e_per_hr"] == pytest.approx(exp, rel=0.3)
    assert q["hot_pixels"] > 0


# ---------------------------------------------------------------- FITS, robustness, speed
def test_fits_round_trip(tmp_path):
    data, hdr, _ = make_frame(_req(MIKE, source=Source(16.0), seed=11))
    path = tmp_path / "mike.fits"
    write_fits(path, data, dict(hdr, OBJECT="HD 12345", EXPSN=50.0))
    d2, h2 = read_fits(path)
    assert d2.dtype == np.uint16 and np.array_equal(d2, data)
    for k in ("GAIN", "OBSTYPE", "ECHG", "RNEFF", "BINNING", "INSTCFG"):
        assert h2[k] == hdr[k]
    assert h2["OBJECT"] == "HD 12345"
    cfg = get_config("MIKE-RED")
    assert quicklook(d2, h2, cfg)["snr_ref_per_A"] == pytest.approx(quicklook(data, hdr, cfg)["snr_ref_per_A"])


def test_quicklook_odd_input():
    cfg = get_config("LRIS-R400")
    q = quicklook(np.zeros((5, 5), dtype=np.uint16), {}, cfg)
    assert q["flags"] and not q["trace_found"]
    data, hdr, _ = make_frame(_req(LRIS, source=Source(18.0), seed=1))
    q = quicklook(data[:, 200:], dict(hdr), cfg)  # cropped frame: generic linear model
    assert any("layout" in f for f in q["flags"])
    q = quicklook(data, {}, cfg)  # no header at all
    assert isinstance(q["flags"], list)
    q = quicklook(data, hdr, get_config("MIKE-RED"))  # wrong instrument
    assert isinstance(q["flags"], list) and q["flags"]


def test_performance():
    for which in (LRIS, MIKE):
        req = _req(which, source=Source(18.0), seed=1)
        make_frame(req)
        t0 = time.perf_counter()
        data, hdr, _ = make_frame(req)
        t_make = time.perf_counter() - t0
        cfg = get_config(which[0])
        quicklook(data, hdr, cfg)
        t0 = time.perf_counter()
        quicklook(data, hdr, cfg)
        t_ql = time.perf_counter() - t0
        assert t_make < 1.5 and t_ql < 1.5  # budget 0.5 / 0.3 s; loose for CI
