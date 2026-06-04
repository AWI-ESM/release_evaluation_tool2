"""IPCC AR6/AR7 colormaps for release_evaluation_tool2.

Loads colormaps from the vendored IPCC-WG1/colormaps repository
(cmpitool/cmpitool/data/ipcc_colormaps/continuous_colormaps_rgb_0-1/),
keeping a single copy of the data shared with cmpitool.
"""

import os
import numpy as np
import matplotlib.colors as mcolors

# Bundled subset of the IPCC AR6/AR7 colormap library (continuous,
# rgb 0-1, ~150 KB). Lives in bg_routines/ rather than under the
# cmpitool submodule so reval does not depend on a particular
# cmpitool revision.
_DATA_DIR = os.path.join(os.path.dirname(__file__), "ipcc_colormaps_data")

_cache = {}


def get_ipcc_cmap(name):
    """Return a matplotlib LinearSegmentedColormap for an IPCC colormap.

    Append '_r' to reverse (e.g. 'temp_div_r').
    """
    reverse = name.endswith("_r")
    base = name[:-2] if reverse else name
    key = (base, reverse)
    if key in _cache:
        return _cache[key]
    rgb = np.loadtxt(os.path.join(_DATA_DIR, base + ".txt"))
    if reverse:
        rgb = rgb[::-1]
    cmap = mcolors.LinearSegmentedColormap.from_list("ipcc_" + name, rgb)
    _cache[key] = cmap
    return cmap


# Reval variable -> IPCC colormap mapping. Diverging cmaps for bias-style
# plots, sequential cmaps for absolute fields. Variable names follow the
# IFS/echam short-name convention used by the part scripts.
_BIAS_CMAP_BY_VAR = {
    "2t":     "temp_div",
    "t":      "temp_div",
    "tas":    "temp_div",
    "thetao": "temp_div",
    "temp":   "temp_div",
    "pr":     "prec_div",
    "tp":     "prec_div",
    "lsp":    "prec_div",
    "cp":     "prec_div",
    "uas":    "wind_div",
    "vas":    "wind_div",
    "ua":     "wind_div",
    "u":      "wind_div",
    "siconc": "cryo_div",
    "a_ice":  "cryo_div",
    "m_ice":  "cryo_div",
    "MLD2":   "misc_div",
    "MLD3":   "misc_div",
    "zos":    "slev_div",
    "ssh":    "slev_div",
    "mlotst": "misc_div",
    "so":     "misc_div",
    "salt":   "misc_div",
    "clt":    "misc_div",
    "tcc":    "misc_div",
    "rlut":   "misc_div",
    "ttr":    "misc_div",
    "ssr":    "temp_div",
    "str":    "temp_div",
    "tsr":    "temp_div",
    "ssrd":   "temp_div",
    "zg":     "misc_div",
}

_ABS_CMAP_BY_VAR = {
    "2t":     "temp_seq",
    "t":      "temp_seq",
    "tas":    "temp_seq",
    "thetao": "temp_seq",
    "temp":   "temp_seq",
    "pr":     "prec_seq",
    "tp":     "prec_seq",
    "lsp":    "prec_seq",
    "cp":     "prec_seq",
    "siconc": "cryo_seq",
    "a_ice":  "cryo_seq",
    "m_ice":  "cryo_seq",
    "MLD2":   "misc_seq_1",
    "MLD3":   "misc_seq_1",
    "u":      "wind_seq",
    "ua":     "wind_seq",
    "zos":    "slev_seq",
    "ssh":    "slev_seq",
    # Land-surface fields use vegetation-green prec_seq (closest IPCC
    # equivalent to YlGn). Soil carbon uses chem_seq (brown ramp,
    # IPCC's biogeochemistry palette).
    "lai":      "prec_seq",
    "cover":    "prec_seq",
    "cpools":   "chem_seq",
    "carbon":   "chem_seq",
    "veg":      "prec_seq",
    # Cavity / cryosphere overlay fields.
    "velocity": "wind_seq",
    "speed":    "wind_seq",
    # Mesh resolution overview.
    "mesh_res": "misc_seq_3",
    # Gregory-plot scatter coloured by time progression.
    "time":     "temp_seq",
}

# Bias / diverging extras for variables not in _BIAS_CMAP_BY_VAR above.
_BIAS_EXTRA = {
    "moc":    "wind_div",  # meridional overturning stream function
    "w":      "wind_div",
}


def get_bias_cmap(var_name):
    """Return the IPCC diverging colormap for a bias plot of `var_name`."""
    if var_name in _BIAS_CMAP_BY_VAR:
        return get_ipcc_cmap(_BIAS_CMAP_BY_VAR[var_name])
    if var_name in _BIAS_EXTRA:
        return get_ipcc_cmap(_BIAS_EXTRA[var_name])
    return get_ipcc_cmap("misc_div")


# Line-plot anchor colours derived from the IPCC diverging palettes, for
# scripts that plot time series rather than maps (ENSO indices, T2M
# anomalies, radiation budget). Picking the deep-end of `temp_div` gives
# good contrast against white axes.
IPCC_LINE = {
    # Sampled ~80% in from each end of `temp_div`, not the very last
    # row — the palette ends are nearly black and overwhelm a line
    # plot. These are the standard ColorBrewer RdBu mid-anchors that
    # IPCC AR6 uses for warming/cooling scenarios.
    "warm":     "#b2182b",
    "cool":     "#2166ac",
    "neutral":  "#666666",   # mid-grey — residual / Δ lines
    "obs":      "black",     # reference observations stay black
}


def get_abs_cmap(var_name):
    """Return the IPCC sequential colormap for an absolute-value plot."""
    return get_ipcc_cmap(_ABS_CMAP_BY_VAR.get(var_name, "misc_seq_1"))
