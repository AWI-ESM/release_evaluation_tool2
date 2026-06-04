# Porting release_evaluation_tool2 to AWI-ESM2 (PI_wisofix_c)

## Target simulation

Source (read-only for us): `/work/ba1066/a270107/esm_tools/EXP/PI_wisofix_c/`

We operate on experiments owned by other users, so the tool must not
require write access to the source. Pattern used here: a workspace
mirror under `/work/ab0246/a270092/runtime/PI_wisofix_c/outdata/` that
symlinks `fesom/`, `echam/`, `jsbach/`, `hdmodel/`, `oasis3mct/` from
the source and adds writable dirs (`echam_remapped/`, `oifs ->
echam_remapped`) for preprocessor output. The config's `spinup_path`
points at this workspace, not the source.

- Years 5369–6004 (636 years of output)
- Atmosphere: **echam6** (monthly streams: `_echam`, `_wiso`, `_accw`, `_accw_wiso`, `_ism`, `_co2`)
- Land: **jsbach** (monthly streams: `_jsbach`, `_js_wiso`, `_land`, `_la_wiso`, `_surf`, `_veg`, `_yasso`)
- Ocean/ice: **FESOM** on **core2** mesh (per-year `<var>.fesom.<year>.nc`)
- No oifs, no lpjg

`_veg` carries PFT cover fractions; `_yasso` carries the soil organic
matter decomposition pools (jsbach delegates SOM to yasso, so the
`_jsbach` stream has only boxed/aggregated pools). Phase 4 must
discover variables across `_jsbach`, `_veg`, `_yasso`, `_land`,
`_surf` with `cdo showname` before coding.

## Year assignment (per user)

| Block | Years | Length |
|---|---|---|
| spinup | 5369–5839 | 471 y |
| pi_ctrl | 5840–6004 | 165 y |
| historic | 5840–6004 | 165 y (placeholder, replaced later) |

## Phases

### Phase 1 — Config + FESOM-only smoke test

Create `configs/AWI-ESM2-PI_wisofix_c.py`:

- `spinup_path` = `/work/ba1066/a270107/esm_tools/EXP/PI_wisofix_c/outdata/`
- `pi_ctrl_path` = `historic_path` = `spinup_path` (same simulation)
- year ranges per the table above; `historic_last25y_*` = 5980–6004
- `mesh_name='core2'`, `meshpath` = TBD (likely a standard core2 mesh dir under `/work/ab0246/a270092/input/fesom2/` or similar — needs lookup)
- `reanalysis`, `remap_resolution`, `dpi`, `accumulation_period` — copy from AWI-ESM3 config
- `reference_path` / `reference_years` — TBD (core2-compatible climatology)

**Run only the FESOM-based parts to verify the config wires up.** Disable everything else in `reval.py`'s `SCRIPTS` dict for the first run:

- Enabled: part1 (mesh_plot), part3 (hovm_temp), part5 (sea_ice_thickness), part6 (ice_conc_timeseries), part7 (mld), part13 (fesom_temp_bias), part14 (fesom_salt_bias), part17 (moc), part19 (ocean_temp_sections), part23 (ice_cavity_velocities)
- Disabled: everything else for now

**Acceptance**: ≥9/10 FESOM scripts COMPLETED.

> **Note on the historic placeholder.** With `historic = pi_ctrl` (same
> path, same years), any script that plots `historic − pi_ctrl` will
> produce all-zero diffs. Acceptable as a stand-in until real historic
> data lands; revisit and disable specific diff plots if the zero
> panels are confusing.

### Phase 2 — echam → `atm_remapped` preprocessor

All atmosphere scripts read `<spinup_path>/oifs/atm_remapped_1m_<var>_1m_YYYY-YYYY.nc` (CMOR-ish monthly per-year files). Rather than rewrite the scripts, write a preprocessor that produces files in that exact layout from echam6 native output via `cdo`.

Reference: `preprocessing_examples/preprocess_AWI-CM3-XIOS_monthly.sh` (oifs→same layout) and `cmpitool/preprocessing_examples/noncmore_preprocess_AWI-ESM2.sh` (echam→cmpitool). **Neither covers the 3D winds / geopotential path** (AWI-CM3 XIOS emits gridded fields directly; AWI-ESM2 echam6 does not). Build that piece from scratch.

The preprocessor splits into **two code paths**:

**2a. Surface fields (cheap)** — `_echam` 2D variables, plus accumulation handling.

| Variable | Source stream / name | Used by |
|---|---|---|
| t2m | `_echam` (`temp2`) | part8, part16, part20 |
| tcc / clt | `_echam` (`aclcov`) | part10 |
| tp / pr | `_echam` (`aprl + aprc`) | part18 |
| sf | `_echam` (`aprs`) | part2 |
| ssr, str (surface SW/LW net) | `_echam` (`srads`, `trads`) | part2, part9 |
| sshf, slhf | `_echam` (`ahfs`, `ahfl`) | part2 |
| tsr, ttr (TOA SW/LW net) | `_echam` (`srad0`, `trad0`) | part2, part9, part21 |

**2b. 3D fields (expensive)** — `_echam` stores 3D atmosphere as
**spectral** state (`st` temperature, `svo` vorticity, `sd`
divergence). No `u`, `v`, `geopoth`, or pressure-level fields exist;
`geopoth` is absent from all streams.

| Variable | Derivation | Used by |
|---|---|---|
| ua@300hPa, va | `cdo afterburner` on spectral state: vor/div → u/v, then interp to 300hPa | part11, part12 |
| zg@500hPa | hydrostatic integration of T + surface pressure via `cdo afterburner`, then interp to 500hPa | part11 |

Build a one-year standalone test of 2b before scaling — afterburner config and pressure-level grids are fiddly enough that a clean dry-run is faster than 165-year debugging.

**Accumulation handling.** echam6 surface fluxes are emitted accumulated since output start (the existence of a separate `_accw` stream for accumulated water variables confirms this convention is in play). The preprocessor must divide each flux by seconds-in-the-output-interval before downstream scripts treat them as W/m². Practical recipe: `cdo divc,<seconds_per_month>` after `cdo settaxis` to set a calendar-aware step, or use `cdo deltat` followed by per-step divide. Verify on a single file with `ncdump -h` looking at `cell_methods` and units before committing the recipe.

**`ssr`/`str` mapping note.** `srads`/`trads` in echam6 are net surface SW/LW (downwelling − upwelling). If part2/part9 want down-only or up/down split (CMIP `rsds`/`rsus`/`rlds`/`rlus`), the preprocessor must either (i) emit only `srads`/`trads` and accept the loss of the split, or (ii) compute per-surface-type fluxes (`rsdsiac`, `rsdswac`, etc.) weighted by `friac`/`alake`/cover fractions. Read part2 and part9 first to pick the right one — if they only use the net signal, option (i) is fine.

**Output layout.** Scripts hardcode `'/oifs/'` in their paths. Since
the source experiment is read-only, the symlink lives in the workspace
mirror, not the source:

```bash
WS=/work/ab0246/a270092/runtime/PI_wisofix_c/outdata
mkdir -p "$WS/echam_remapped"
ln -s echam_remapped "$WS/oifs"
```

Preprocessor writes into `$WS/echam_remapped/atm_remapped_1m_<var>_1m_YYYY-YYYY.nc`.

**Acceptance**: 10 atmosphere scripts (part2, part8–12, part16, part18, part20, part21) COMPLETED.

### Phase 3 — part4 (cmpi)

`part4_cmpi.py` shells out to `preprocess_AWI-CM3-XIOS.sh` which is
XIOS-specific, then reruns cmpitool. For PI_wisofix_c the user has
already run cmpitool externally (`cmpitool/run_PI_wisofix_c.py`,
producing `cmpitool/eval/ERA5/PI_wisofix_c.csv`), so the right
end-state is for part4 to consume that csv rather than re-run.

Two approaches:

1. Have the config provide a `cmpitool_csv` override; part4 checks for
   it and short-circuits the preprocess/cmpitool run.
2. Rename `model_version` to `PI_wisofix_c` so the existing csv is
   found by name. (Breaks the prefixed naming convention used by
   `out_path` etc.)

Approach 1 is cleaner. Until that's wired, part4 is disabled in the
AWI-ESM2 config's `scripts_overrides`.

**Acceptance**: part4 COMPLETED via the existing csv.

### Phase 4 — jsbach replacements for part24/25/26

**Variable discovery (done for PI_wisofix_c, from the per-stream
`.codes` files in `outdata/jsbach/`).**

- LAI → `_jsbach`, code **107** `lai`, 11 PFT levels, dimensionless
- Cover fraction → `_jsbach`, code **12** `cover_fract`, 11 PFT levels
  (also `act_fpc` in `_veg` at code 31, both 11-level)
- Carbon pools → `_yasso` only. **There are no `box_Cpools_*` in
  `_jsbach` for this run** (the codes file is just 30 lines, all
  surface-process variables, no aggregated Cpools). Yasso carries 18
  pools: `boxYC_{acid,water,ethanol,nonsoluble}_{ag,bg}{1,2}` (16) plus
  `boxYC_humus_{1,2}` (2), codes 31–49. The `_1` / `_2` suffix splits
  woody vs non-woody litter. **Total terrestrial C = sum of all 18.**

The stream files are GRIB1 on the T63 Gaussian grid (192x96 = 18432
points). Since the run's `.codes` file format is not accepted by
`cdo setpartabn`, the preprocessor / scripts use code numbers directly
(`-selcode,107` etc.) and rename `var<code>` -> human name via
`-chname`.

Port the three LPJ-GUESS scripts:

- **part24_lpjg_lai.py → part24_jsbach_lai.py**: read LAI, plot
  global/seasonal climatology.
- **part25_lpjg_carbon.py → part25_jsbach_carbon.py**: aggregate boxed
  pools from `_jsbach` *and* yasso-resolved litter/SOM pools from
  `_yasso`. Total terrestrial C is the sum across both streams.
- **part26_lpjg_pft.py → part26_jsbach_pft.py**: read `cover_fract`
  from `_veg` (PFT dim depends on jsbach configuration — discover via
  `cdo showname`/`ncdump -h`).

Each follows the same pattern as the LPJG version: `config_loader` →
load mesh/years → cdo to monthly clim → plot to `out_path`. Update
`reval.py`'s `SCRIPTS` dict to swap names.

**Acceptance**: 3 new jsbach scripts COMPLETED.

### Phase 5 — part22 (masks)

Disable in `reval.py`'s `SCRIPTS` dict for the AWI-ESM2 config. The `grid_configs` dict in part22 hardcodes A096/TCO95-land grid names from AWI-ESM3/oifs; AWI-ESM2 uses different OASIS grids. Not worth porting now.

## Open items before Phase 1 starts

1. **Path to core2 mesh** — needs confirmation. Likely candidates to check: `/work/ab0246/a270092/input/fesom2/core2/`, or under `/work/ba1066/`. Look for `nod2d.out`, `elem2d.out`, `aux3d.out`, `fesom.mesh.diag.nc`.
2. **Core2 reference climatology** for part3, part13, part14 (`reference_path` + `reference_years`) — does one exist in `/work/ab0246/a270092/postprocessing/climatologies/`?
3. **Radiation split for part2/part9** (resolved before writing 2a): do those scripts use only `ssr`/`str` net signals, or also need down/up split? Read scripts before picking the preprocessor recipe.

## Final state target

| Block | Scripts | Status target |
|---|---|---|
| FESOM | part1, 3, 5–7, 13, 14, 17, 19, 23 | COMPLETED via config alone |
| ECHAM (preprocessor route) | part2, 8–12, 16, 18, 20, 21 | COMPLETED |
| cmpi | part4 | COMPLETED via vendored cmpitool config |
| jsbach (new) | part24, 25, 26 (renamed) | COMPLETED |
| Disabled | part22 | Skipped |

**Net: 24/26 enabled jobs COMPLETED, part22 intentionally skipped, part17_moc unchanged from AWI-ESM3 run.**
