#!/usr/bin/env python3
"""
Generate an HTML report of all evaluation plots.
Scans the output directory and creates a self-contained HTML file
with embedded base64 images, organized by diagnostic category.

Usage:
    python generate_report.py                         # uses config from config_loader
    python generate_report.py /path/to/output/dir     # explicit output directory
"""

import sys
import os
import base64
import glob
from datetime import datetime
from pathlib import Path
from collections import OrderedDict

# --------------------------------------------------------------------------
# Category definitions: map filename patterns to sections
# Order matters - first match wins
# --------------------------------------------------------------------------
CATEGORIES = OrderedDict([
    ("Mesh & Masks", {
        "patterns": ["mesh_resolution*", "mask_*", "masks_overview*"],
        "description": "Mesh resolution and coupling masks",
    }),
    ("Radiation Balance", {
        "patterns": ["rad_budget*", "radiation*"],
        "description": "Top-of-atmosphere and surface radiation budget",
    }),
    ("Radiation vs CERES", {
        "patterns": ["*crf*ceres*", "*ceres*", "swcrf*", "lwcrf*", "*_vs_CERES*"],
        "description": "Cloud radiative forcing bias compared to CERES-EBAF",
    }),
    ("Cloud Cover", {
        "patterns": ["*clt*", "cloud_cover*", "*_vs_MODIS*", "*zonal_cloud*"],
        "description": "Cloud cover compared to MODIS observations",
    }),
    ("Temperature (2m) vs ERA5", {
        "patterns": ["t2m_*", "T2M_*"],
        "description": "Near-surface temperature compared to ERA5 reanalysis",
    }),
    ("Precipitation vs GPCP", {
        "patterns": ["precip*", "Precip*"],
        "description": "Precipitation compared to GPCP observations",
    }),
    ("Zonal Mean Biases", {
        "patterns": ["*zonal_mean*", "*zonal*"],
        "description": "Zonal mean atmospheric profiles",
    }),
    ("QBO", {
        "patterns": ["*qbo*", "*QBO*"],
        "description": "Quasi-Biennial Oscillation",
    }),
    ("Climate Performance Index", {
        "patterns": ["CMPI*", "cmpi*"],
        "description": "Climate Model Performance Index",
    }),
    ("Hovmoeller Diagrams", {
        "patterns": ["Hovmoeller*", "hovmoeller*", "hovm*"],
        "description": "Hovmoeller diagrams of ocean temperature",
    }),
    ("Ocean Temperature Bias", {
        "patterns": ["temp_bias*", "fesom_temp*", "ocean_temp*", "*temp_section*",
                      "*_A16_*", "*_P16_*"],
        "description": "Ocean temperature biases and sections",
    }),
    ("Salinity Bias", {
        "patterns": ["salt_bias*", "fesom_salt*", "salinity*"],
        "description": "Ocean salinity biases",
    }),
    ("Mixed Layer Depth", {
        "patterns": ["MLD*", "mld*"],
        "description": "Mixed layer depth",
    }),
    ("Sea Ice Thickness", {
        "patterns": ["*sea_ice_thickness*", "GIOMAS*"],
        "description": "Sea ice thickness compared to GIOMAS",
    }),
    ("Sea Ice Extent", {
        "patterns": ["a_ice_*", "sea_ice_extent*", "sea_ice_area*", "ice_conc*"],
        "description": "Sea ice concentration and extent time series",
    }),
    ("ENSO", {
        "patterns": ["*enso*", "*ENSO*", "*Nino*", "*nino*"],
        "description": "ENSO diagnostics and teleconnections",
    }),
    ("AMOC & Overturning", {
        "patterns": ["*amoc*", "*AMOC*", "*moc*", "*overturning*"],
        "description": "Atlantic Meridional Overturning Circulation",
    }),
    ("Gregory Plot & Drift", {
        "patterns": ["*gregory*", "*Gregory*", "*drift*", "*pi_control*"],
        "description": "PI control drift and Gregory plot analysis",
    }),
    ("Climate Change", {
        "patterns": ["*hist-pict*", "*climate_change*", "*warming*"],
        "description": "Historical vs. pre-industrial climate change signals",
    }),
    ("Ice Shelf Cavities", {
        "patterns": ["ice_cavity_*"],
        "description": "Ice shelf cavity circulation and velocities",
    }),
    ("LPJ-GUESS: LAI", {
        "patterns": ["lpjg_LAI*"],
        "description": "Leaf Area Index from LPJ-GUESS",
    }),
    ("LPJ-GUESS: Carbon", {
        "patterns": ["lpjg_Carbon*", "lpjg_SoilCarbon*", "lpjg_EcosystemCarbon*"],
        "description": "Carbon pools from LPJ-GUESS",
    }),
    ("LPJ-GUESS: PFTs", {
        "patterns": ["lpjg_dominant*", "lpjg_PFT*"],
        "description": "Plant Functional Types from LPJ-GUESS",
    }),
])


def img_to_base64(filepath):
    """Read an image file and return a base64-encoded data URI."""
    with open(filepath, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    ext = Path(filepath).suffix.lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "svg": "image/svg+xml", "gif": "image/gif"}.get(ext, "image/png")
    return f"data:{mime};base64,{data}"


def categorize_files(png_files):
    """Assign each PNG to a category. Uncategorized files go to 'Other'."""
    import fnmatch
    categorized = OrderedDict()
    for cat in CATEGORIES:
        categorized[cat] = []
    categorized["Other"] = []

    used = set()
    for cat, info in CATEGORIES.items():
        for png in png_files:
            if png in used:
                continue
            basename = os.path.basename(png)
            for pattern in info["patterns"]:
                if fnmatch.fnmatch(basename, pattern):
                    categorized[cat].append(png)
                    used.add(png)
                    break

    for png in png_files:
        if png not in used:
            categorized["Other"].append(png)

    # Remove empty categories
    return OrderedDict((k, v) for k, v in categorized.items() if v)


def human_filename(filepath):
    """Convert a filename to a human-readable title."""
    name = Path(filepath).stem
    # Clean up common patterns
    name = name.replace("_", " ").replace("-", " — ")
    return name


def generate_html(output_dir, model_name):
    """Generate the HTML report string."""
    png_files = sorted(glob.glob(os.path.join(output_dir, "*.png")))

    if not png_files:
        print(f"No PNG files found in {output_dir}")
        return None

    categories = categorize_files(png_files)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_plots = len(png_files)
    n_sections = len(categories)

    # Build navigation and sections
    nav_items = []
    sections = []
    for cat, files in categories.items():
        cat_id = cat.lower().replace(" ", "-").replace(":", "").replace("&", "and")
        desc = CATEGORIES.get(cat, {}).get("description", "")
        nav_items.append(f'<a href="#{cat_id}" class="nav-link">{cat} <span class="badge">{len(files)}</span></a>')

        img_cards = []
        for f in files:
            b64 = img_to_base64(f)
            title = human_filename(f)
            fname = os.path.basename(f)
            img_cards.append(f'''
                <div class="card">
                    <div class="card-title">{title}</div>
                    <a href="{fname}" target="_blank">
                        <img src="{b64}" alt="{title}" loading="lazy">
                    </a>
                    <div class="card-filename">{fname}</div>
                </div>''')

        sections.append(f'''
        <section id="{cat_id}">
            <h2>{cat}</h2>
            {"<p class='section-desc'>" + desc + "</p>" if desc else ""}
            <div class="grid">
                {"".join(img_cards)}
            </div>
        </section>''')

    nav_html = "\n".join(nav_items)
    sections_html = "\n".join(sections)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{model_name} — Release Evaluation</title>
<style>
:root {{
    --bg: #f8f9fa;
    --sidebar-bg: #1a1d23;
    --sidebar-text: #c9d1d9;
    --sidebar-active: #58a6ff;
    --card-bg: #ffffff;
    --text: #24292f;
    --text-muted: #656d76;
    --border: #d0d7de;
    --accent: #0969da;
    --section-bg: #ffffff;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    display: flex;
    min-height: 100vh;
}}
/* Sidebar */
.sidebar {{
    position: fixed;
    top: 0; left: 0;
    width: 280px;
    height: 100vh;
    background: var(--sidebar-bg);
    color: var(--sidebar-text);
    overflow-y: auto;
    padding: 24px 0;
    z-index: 100;
}}
.sidebar-header {{
    padding: 0 20px 20px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 16px;
}}
.sidebar-header h1 {{
    font-size: 16px;
    color: #fff;
    font-weight: 600;
    line-height: 1.4;
}}
.sidebar-header .meta {{
    font-size: 12px;
    color: #8b949e;
    margin-top: 8px;
}}
.nav-link {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 20px;
    color: var(--sidebar-text);
    text-decoration: none;
    font-size: 13px;
    transition: all 0.15s;
    border-left: 3px solid transparent;
}}
.nav-link:hover {{
    background: rgba(255,255,255,0.05);
    color: #fff;
    border-left-color: var(--sidebar-active);
}}
.badge {{
    background: rgba(255,255,255,0.1);
    color: #8b949e;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 500;
}}
/* Main content */
.main {{
    margin-left: 280px;
    flex: 1;
    padding: 40px;
    max-width: 1400px;
}}
.main > header {{
    margin-bottom: 40px;
    padding-bottom: 24px;
    border-bottom: 1px solid var(--border);
}}
.main > header h1 {{
    font-size: 28px;
    font-weight: 700;
    color: var(--text);
}}
.main > header p {{
    color: var(--text-muted);
    margin-top: 8px;
    font-size: 14px;
}}
section {{
    margin-bottom: 48px;
    scroll-margin-top: 24px;
}}
section h2 {{
    font-size: 20px;
    font-weight: 600;
    color: var(--text);
    padding-bottom: 12px;
    border-bottom: 2px solid var(--accent);
    margin-bottom: 8px;
}}
.section-desc {{
    color: var(--text-muted);
    font-size: 14px;
    margin-bottom: 16px;
}}
.grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(480px, 1fr));
    gap: 20px;
}}
.card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    transition: box-shadow 0.2s;
}}
.card:hover {{
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
}}
.card-title {{
    padding: 12px 16px 8px;
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
}}
.card img {{
    width: 100%;
    height: auto;
    display: block;
    cursor: pointer;
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
}}
.card-filename {{
    padding: 6px 16px 10px;
    font-size: 11px;
    color: var(--text-muted);
    font-family: "SFMono-Regular", Consolas, monospace;
}}
/* Back to top */
.back-to-top {{
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: var(--accent);
    color: #fff;
    width: 40px; height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    text-decoration: none;
    font-size: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    opacity: 0.7;
    transition: opacity 0.2s;
}}
.back-to-top:hover {{ opacity: 1; }}
/* Responsive */
@media (max-width: 900px) {{
    .sidebar {{ display: none; }}
    .main {{ margin-left: 0; padding: 20px; }}
    .grid {{ grid-template-columns: 1fr; }}
}}
/* Lightbox */
.lightbox {{
    display: none;
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.85);
    z-index: 1000;
    cursor: pointer;
    align-items: center;
    justify-content: center;
}}
.lightbox.active {{ display: flex; }}
.lightbox img {{
    max-width: 95vw;
    max-height: 95vh;
    border-radius: 4px;
    box-shadow: 0 0 40px rgba(0,0,0,0.5);
}}
</style>
</head>
<body>

<nav class="sidebar">
    <div class="sidebar-header">
        <h1>{model_name}</h1>
        <div class="meta">Release Evaluation Report<br>{timestamp}<br>{n_plots} plots in {n_sections} sections</div>
    </div>
    {nav_html}
</nav>

<div class="main">
    <header>
        <h1>{model_name} — Release Evaluation</h1>
        <p>Generated {timestamp} &mdash; {n_plots} diagnostic plots</p>
    </header>
    {sections_html}
</div>

<a href="#" class="back-to-top" title="Back to top">&uarr;</a>

<div class="lightbox" id="lightbox" onclick="this.classList.remove('active')">
    <img id="lightbox-img" src="" alt="">
</div>

<script>
document.querySelectorAll('.card img').forEach(img => {{
    img.addEventListener('click', e => {{
        e.preventDefault();
        e.stopPropagation();
        document.getElementById('lightbox-img').src = img.src;
        document.getElementById('lightbox').classList.add('active');
    }});
}});
document.addEventListener('keydown', e => {{
    if (e.key === 'Escape') document.getElementById('lightbox').classList.remove('active');
}});
// Highlight active nav on scroll
const sections = document.querySelectorAll('section');
const navLinks = document.querySelectorAll('.nav-link');
const observer = new IntersectionObserver(entries => {{
    entries.forEach(entry => {{
        if (entry.isIntersecting) {{
            navLinks.forEach(link => {{
                link.style.borderLeftColor = link.getAttribute('href') === '#' + entry.target.id ? 'var(--sidebar-active)' : 'transparent';
                link.style.color = link.getAttribute('href') === '#' + entry.target.id ? '#fff' : '';
            }});
        }}
    }});
}}, {{ threshold: 0.2 }});
sections.forEach(s => observer.observe(s));
</script>

</body>
</html>'''
    return html


def main():
    # Determine output directory
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
        model_name = os.path.basename(output_dir.rstrip("/"))
    else:
        # Use config
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from bg_routines.config_loader import out_path, model_version
        output_dir = out_path
        model_name = model_version

    output_dir = os.path.abspath(output_dir)
    print(f"Scanning: {output_dir}")
    print(f"Model:    {model_name}")

    html = generate_html(output_dir, model_name)
    if html is None:
        sys.exit(1)

    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, "w") as f:
        f.write(html)

    size_mb = os.path.getsize(report_path) / (1024 * 1024)
    print(f"\nReport saved: {report_path} ({size_mb:.1f} MB)")
    print(f"Open in browser or via JupyterHub file browser.")


if __name__ == "__main__":
    main()
