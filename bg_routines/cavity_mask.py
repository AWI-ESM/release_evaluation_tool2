"""
Cavity mask utilities for FESOM2 meshes.

Identifies ice shelf cavity elements by comparing elvls.out (actual active
levels per element, accounting for ice shelf draft) with elvls_raw.out
(levels based on bottom topography only). Elements where these differ have
their top levels blocked by an ice shelf.
"""

import numpy as np
import os
import re


def check_use_cavity(spinup_path):
    """
    Check if the FESOM2 simulation was run with ice shelf cavities.
    
    Reads use_cavity from <runtime_dir>/config/fesom/namelist.config.
    The runtime directory is derived from spinup_path by going up from outdata/.
    
    Parameters
    ----------
    spinup_path : str
        Path to the simulation output (e.g. .../outdata/ or .../outdata/fesom/)
    
    Returns
    -------
    bool
        True if use_cavity = .true. in the namelist, False otherwise
    """
    # Walk up from spinup_path to find config/fesom/namelist.config
    path = os.path.abspath(spinup_path.rstrip('/'))
    for _ in range(5):  # Try up to 5 levels up
        namelist = os.path.join(path, 'config', 'fesom', 'namelist.config')
        if os.path.exists(namelist):
            break
        path = os.path.dirname(path)
    else:
        print(f"  WARNING: namelist.config not found, assuming no cavities")
        return False
    
    with open(namelist, 'r') as f:
        content = f.read()
    
    match = re.search(r'use_cavity\s*=\s*\.(true|false)\.', content, re.IGNORECASE)
    if match:
        result = match.group(1).lower() == 'true'
        print(f"  use_cavity = .{match.group(1)}.  (from {namelist})")
        return result
    
    print(f"  WARNING: use_cavity not found in {namelist}, assuming false")
    return False


def get_cavity_element_mask(mesh, meshpath):
    """
    Determine which elements are inside ice shelf cavities.
    
    Priority:
      1. mesh.nc 'cav_elem_lev' variable (most reliable)
      2. elvls.out vs elvls_raw.out comparison (fallback)
    
    Parameters
    ----------
    mesh : pyfesom2 mesh object
    meshpath : str
        Path to the FESOM2 mesh directory
    
    Returns
    -------
    elem_cavity : np.ndarray of bool, shape (n_elements,)
        True for cavity elements, False for open ocean elements
    """
    meshpath = meshpath.rstrip('/')
    mesh_nc = os.path.join(meshpath, 'mesh.nc')
    
    # Try mesh.nc first (has proper cavity info)
    if os.path.exists(mesh_nc):
        try:
            from netCDF4 import Dataset
            ds = Dataset(mesh_nc, 'r')
            if 'cav_elem_lev' in ds.variables:
                cel = ds.variables['cav_elem_lev'][:]
                elem_cavity = np.array(cel > 0)
                n_cav = int(np.sum(elem_cavity))
                print(f"  Cavity elements (from mesh.nc cav_elem_lev): {n_cav} / {len(elem_cavity)} "
                      f"({100*n_cav/len(elem_cavity):.1f}%)")
                ds.close()
                return elem_cavity
            ds.close()
        except Exception as e:
            print(f"  WARNING: Failed to read cavity info from mesh.nc: {e}")
    
    # Fallback: elvls.out vs elvls_raw.out
    elvls_file = os.path.join(meshpath, 'elvls.out')
    elvls_raw_file = os.path.join(meshpath, 'elvls_raw.out')
    
    if not os.path.exists(elvls_raw_file):
        print(f"  WARNING: No cavity info found in mesh, cannot detect cavities")
        return np.zeros(len(mesh.elem), dtype=bool)
    
    elvls = np.loadtxt(elvls_file, dtype=int)
    elvls_raw = np.loadtxt(elvls_raw_file, dtype=int)
    elem_cavity = elvls != elvls_raw
    
    n_cav = int(np.sum(elem_cavity))
    print(f"  Cavity elements (from elvls): {n_cav} / {len(elem_cavity)} "
          f"({100*n_cav/len(elem_cavity):.1f}%)")
    
    return elem_cavity


def get_cavity_node_mask(mesh, meshpath):
    """
    Determine which nodes are inside ice shelf cavities.
    
    Priority:
      1. mesh.nc 'cav_nod_mask' variable (most reliable)
      2. Derived from element cavity mask (fallback)
    
    Parameters
    ----------
    mesh : pyfesom2 mesh object
    meshpath : str
        Path to the FESOM2 mesh directory
    
    Returns
    -------
    node_cavity : np.ndarray of bool, shape (n_nodes,)
        True for cavity nodes, False for open ocean nodes
    """
    meshpath = meshpath.rstrip('/')
    mesh_nc = os.path.join(meshpath, 'mesh.nc')
    
    # Try mesh.nc first
    if os.path.exists(mesh_nc):
        try:
            from netCDF4 import Dataset
            ds = Dataset(mesh_nc, 'r')
            if 'cav_nod_mask' in ds.variables:
                cnm = ds.variables['cav_nod_mask'][:]
                node_cavity = np.array(cnm == 1)
                n_cav = int(np.sum(node_cavity))
                print(f"  Cavity nodes (from mesh.nc cav_nod_mask): {n_cav} / {len(node_cavity)} "
                      f"({100*n_cav/len(node_cavity):.1f}%)")
                ds.close()
                return node_cavity
            ds.close()
        except Exception as e:
            print(f"  WARNING: Failed to read cavity node info from mesh.nc: {e}")
    
    # Fallback: derive from element mask
    elem_cavity = get_cavity_element_mask(mesh, meshpath)
    node_cavity = np.zeros(mesh.n2d, dtype=bool)
    cavity_elem_indices = np.where(elem_cavity)[0]
    for idx in cavity_elem_indices:
        node_cavity[mesh.elem[idx]] = True
    
    n_cav = int(np.sum(node_cavity))
    print(f"  Cavity nodes (from elements): {n_cav} / {mesh.n2d} "
          f"({100*n_cav/mesh.n2d:.1f}%)")
    
    return node_cavity
