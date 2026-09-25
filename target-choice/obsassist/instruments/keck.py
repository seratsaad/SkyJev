"""Keck instrument configurations.

Sourced values (URLs in docs/research/keck_parameters.json): detector read noise, gain, dark
current, pixel scales, dispersions and wall-clock readout times from the instrument pages;
throughput: DEIMOS measured 2018 (incl. telescope), MOSFIRE measured per band (incl. telescope),
LRIS-B instrument efficiency times the telescope (primary 0.86 x secondary 0.88 x dichroic 0.95);
LRIS-R 400/8500 total (0.30 peak) and HIRES total (0.07 peak) are estimates. Acquisition:
~5 min typical (Keck guide), ~10 min for faint long-slit targets (estimate).
"""

from obsassist.instruments.base import InstrumentConfig, register

K1 = ("keck1",)
K2 = ("keck2",)
SRC = ("docs/research/keck_parameters.json",)
_TEL = 0.86 * 0.88 * 0.95  # primary x secondary x dichroic (LRIS loss budget)

# ---------------------------------------------------------------- LRIS (Keck I Cassegrain)
register(
    InstrumentConfig(
        key="LRIS-B600",
        instrument="LRIS",
        telescopes=K1,
        mode="spec",
        setup="LRIS-LS-D560",
        description='LRIS blue arm, 600/4000 grism, 1.0" long slit, D560 dichroic, 1x1',
        lam_min=3150.0,
        lam_max=5600.0,
        lam_ref=4500.0,
        throughput=(
            (3150, 0.05),
            (3500, 0.22 * _TEL),
            (4000, 0.43 * _TEL),
            (4221, 0.49 * _TEL),
            (4500, 0.48 * _TEL),
            (5000, 0.445 * _TEL),
            (5400, 0.40 * _TEL),
            (5600, 0.12),
        ),
        pixscale=0.135,
        read_noise=3.8,
        dark_e_per_hr=2.0,
        gain=1.6,
        full_well=99000.0,
        readout_s=54.0,
        dispersion=0.63,
        resolution=1100.0,
        slit_width=1.0,
        slit_length=168.0,
        max_exp_s=1800.0,
        acq_s=480.0,
        config_change_s=180.0,
        adc=True,
        readout_speeds={"slow": 54.0, "fast": 36.0},
        confidence="documented",
        sources=SRC,
    )
)
register(
    InstrumentConfig(
        key="LRIS-R400",
        instrument="LRIS",
        telescopes=K1,
        mode="spec",
        setup="LRIS-LS-D560",
        description='LRIS red arm, 400/8500 grating, 1.0" long slit, D560 dichroic, 4-amp 1x1',
        lam_min=5600.0,
        lam_max=10000.0,
        lam_ref=7000.0,
        throughput=((5600, 0.10), (6000, 0.24), (7000, 0.30), (8000, 0.27), (9000, 0.18), (10000, 0.06)),
        pixscale=0.135,
        read_noise=3.6,
        dark_e_per_hr=5.0,
        gain=1.66,
        full_well=98000.0,
        readout_s=54.0,
        dispersion=1.16,
        resolution=1000.0,
        slit_width=1.0,
        slit_length=168.0,
        max_exp_s=1200.0,
        acq_s=480.0,
        config_change_s=180.0,
        adc=True,
        readout_speeds={"slow": 54.0, "fast": 24.0},
        confidence="approximate",
        sources=SRC,
    )
)
register(
    InstrumentConfig(
        key="LRIS-IMG-R",
        instrument="LRIS",
        telescopes=K1,
        mode="img",
        setup="LRIS-IMG",
        description="LRIS red-side imaging, R filter",
        lam_min=5700.0,
        lam_max=7300.0,
        lam_ref=6500.0,
        band="R",
        band_width=1300.0,
        throughput=((5700, 0.30), (6500, 0.38), (7300, 0.30)),
        pixscale=0.135,
        read_noise=3.6,
        dark_e_per_hr=5.0,
        gain=1.66,
        full_well=98000.0,
        readout_s=54.0,
        max_exp_s=600.0,
        acq_s=180.0,
        config_change_s=180.0,
        adc=True,
        confidence="approximate",
        sources=SRC,
    )
)

# ---------------------------------------------------------------- MOSFIRE (Keck I Cassegrain)
# sky between OH lines relative to the band-average sky (MOSFIRE measured): J ~0.11, H ~0.06, K ~0.15
_MOS = dict(
    instrument="MOSFIRE",
    telescopes=K1,
    pixscale=0.1798,
    gain=2.15,
    full_well=40000.0,
    dark_e_per_hr=28.8,
    read_noise=5.8,
    readout_s=11.0,
    slit_width=0.7,
    slit_length=46.0,
    acq_s=600.0,
    config_change_s=120.0,
    nir=True,
    adc=False,
    confidence="documented",
    sources=SRC,
)
register(
    InstrumentConfig(
        key="MOSFIRE-J",
        mode="spec",
        setup="MOSFIRE-J",
        description='MOSFIRE J long slit 0.7", MCDS16, ABBA',
        lam_min=11530.0,
        lam_max=13520.0,
        lam_ref=12500.0,
        sky_line_factor=0.11,
        throughput=((11530, 0.18), (12000, 0.28), (12500, 0.32), (13000, 0.28), (13520, 0.15)),
        dispersion=1.3028,
        resolution=3310.0,
        max_exp_s=120.0,
        **_MOS,
    )
)
register(
    InstrumentConfig(
        key="MOSFIRE-H",
        mode="spec",
        setup="MOSFIRE-H",
        description='MOSFIRE H long slit 0.7", MCDS16, ABBA',
        lam_min=14680.0,
        lam_max=18040.0,
        lam_ref=16350.0,
        sky_line_factor=0.06,
        throughput=((14680, 0.22), (15500, 0.34), (16350, 0.38), (17200, 0.34), (18040, 0.20)),
        dispersion=1.6269,
        resolution=3660.0,
        max_exp_s=120.0,
        **_MOS,
    )
)
register(
    InstrumentConfig(
        key="MOSFIRE-K",
        mode="spec",
        setup="MOSFIRE-K",
        description='MOSFIRE K long slit 0.7", MCDS16, ABBA',
        lam_min=19540.0,
        lam_max=23970.0,
        lam_ref=21900.0,
        sky_line_factor=0.15,
        throughput=((19540, 0.20), (20500, 0.31), (21900, 0.36), (23000, 0.30), (23970, 0.18)),
        dispersion=2.1691,
        resolution=3620.0,
        max_exp_s=180.0,
        **_MOS,
    )
)

# ---------------------------------------------------------------- HIRES (Keck I Nasmyth)
register(
    InstrumentConfig(
        key="HIRES-C2",
        instrument="HIRES",
        telescopes=K1,
        mode="spec",
        setup="HIRES-C2",
        description='HIRES, C2 decker (0.861" x 14"), red cross-disperser, 2x1 binning, low gain',
        lam_min=3600.0,
        lam_max=8000.0,
        lam_ref=5500.0,
        throughput=((3600, 0.015), (4000, 0.03), (5000, 0.06), (5500, 0.07), (6500, 0.07), (8000, 0.04)),
        pixscale=0.12,
        read_noise=3.0,
        dark_e_per_hr=2.0,
        gain=2.0,
        full_well=78000.0,
        readout_s=40.0,
        dispersion=0.024,
        resolution=48000.0,
        slit_width=0.861,
        slit_length=14.0,
        bin_spatial=2,
        max_exp_s=1800.0,
        acq_s=240.0,
        config_change_s=300.0,
        adc=False,
        readout_speeds={"1x1": 60.0, "2x1": 40.0, "2x2": 24.0},
        confidence="approximate",
        sources=SRC,
    )
)

# ---------------------------------------------------------------- DEIMOS (Keck II Nasmyth)
_DEI = dict(
    instrument="DEIMOS",
    telescopes=K2,
    pixscale=0.1185,
    gain=1.22,
    full_well=79000.0,
    dark_e_per_hr=3.5,
    read_noise=2.55,
    readout_s=80.0,
    max_exp_s=1800.0,
    adc=False,
    confidence="documented",
    sources=SRC,
)
register(
    InstrumentConfig(
        key="DEIMOS-600ZD",
        mode="spec",
        setup="DEIMOS-600ZD",
        description='DEIMOS 600ZD grating, GG455, 1.0" long slit / slitmask (measured 2018 throughput)',
        lam_min=4550.0,
        lam_max=9600.0,
        lam_ref=7000.0,
        throughput=(
            (4550, 0.06),
            (5000, 0.127),
            (6000, 0.213),
            (7000, 0.244),
            (7400, 0.246),
            (8000, 0.233),
            (8500, 0.206),
            (9000, 0.15),
            (9600, 0.07),
        ),
        dispersion=0.65,
        resolution=2100.0,
        slit_width=1.0,
        slit_length=16.0,
        acq_s=600.0,
        config_change_s=240.0,
        **_DEI,
    )
)
register(
    InstrumentConfig(
        key="DEIMOS-1200G",
        mode="spec",
        setup="DEIMOS-1200G",
        description='DEIMOS 1200G grating, OG550, 0.7" slitmask',
        lam_min=6400.0,
        lam_max=9100.0,
        lam_ref=8500.0,
        throughput=((6400, 0.10), (7000, 0.17), (7800, 0.20), (8500, 0.19), (9100, 0.12)),
        dispersion=0.33,
        resolution=6000.0,
        slit_width=0.7,
        slit_length=8.0,
        acq_s=900.0,
        config_change_s=240.0,
        **_DEI,
    )
)
