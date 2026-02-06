import xml.etree.ElementTree as ET
import numpy as np
import os
from pathlib import Path

def parse_svg_experiment(svg_path, output_dir, scale=1.0, z_layer=0.0):
    """
    Parses an SVG file to generate a Demlat experiment.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    
    nodes = []
    node_colors = []
    bars = []
    bar_colors = []
    
    # Helper to parse color
    def parse_color(style_str):
        if not style_str: return None
        
        # Case 1: Direct hex code (e.g. "#ff0000")
        if style_str.startswith('#'):
            return style_str.lower()
            
        # Case 2: CSS Style string (e.g. "fill:#ff0000; stroke:none")
        parts = style_str.split(';')
        for p in parts:
            if 'stroke:' in p or 'fill:' in p:
                try:
                    val = p.split(':')[1].strip()
                    if val.startswith('#'):
                        return val.lower()
                except IndexError:
                    pass
        return None

    # 1. Extract Nodes (Circles)
    for circle in root.findall('.//svg:circle', ns):
        cx = float(circle.get('cx')) * scale
        cy = float(circle.get('cy')) * scale
        
        style = circle.get('style')
        if not style:
            style = circle.get('fill')
            
        color = parse_color(style)
        
        nodes.append([cx, -cy, z_layer]) # Flip Y for standard coord system
        node_colors.append(color)

    # 2. Extract Bars (Lines)
    lines = root.findall('.//svg:line', ns)
    node_arr = np.array(nodes)
    
    for line in lines:
        x1 = float(line.get('x1')) * scale
        y1 = -float(line.get('y1')) * scale
        x2 = float(line.get('x2')) * scale
        y2 = -float(line.get('y2')) * scale
        
        style = line.get('style')
        if not style:
            style = line.get('stroke')
            
        color = parse_color(style)
        
        # Find closest nodes
        p1 = np.array([x1, y1, z_layer])
        p2 = np.array([x2, y2, z_layer])
        
        d1 = np.linalg.norm(node_arr - p1, axis=1)
        d2 = np.linalg.norm(node_arr - p2, axis=1)
        
        idx1 = np.argmin(d1)
        idx2 = np.argmin(d2)
        
        if d1[idx1] < 1e-3 and d2[idx2] < 1e-3 and idx1 != idx2:
            bars.append([idx1, idx2])
            bar_colors.append(color)

    # 3. Process Attributes
    n_nodes = len(nodes)
    mass = np.ones(n_nodes, dtype=float)
    attributes = np.zeros(n_nodes, dtype=int) # 0: Free
    
    # Attribute Bitmasks (matching solver)
    ATTR_FIXED = 1
    ATTR_POS_DRIVEN = 2
    
    actuators = []
    fixed_indices = []
    
    for i, color in enumerate(node_colors):
        if color == '#ff0000': # Red -> Actuated
            attributes[i] = ATTR_POS_DRIVEN
            actuators.append({
                'node_idx': i,
                'type': 'position', 
                'signal_name': f'sig_node_{i}'
            })
        elif color == '#000000': # Black -> Fixed
            attributes[i] = ATTR_FIXED
            mass[i] = 0.0
            fixed_indices.append(i)

    print(f"[Importer] Found {len(fixed_indices)} fixed nodes: {fixed_indices}")

    # 4. Process Bars & Stiffness
    n_bars = len(bars)
    bar_indices = np.array(bars, dtype=int)
    stiffness = np.zeros(n_bars, dtype=float)
    rest_lengths = np.zeros(n_bars, dtype=float)
    damping = np.zeros(n_bars, dtype=float)
    
    for i, color in enumerate(bar_colors):
        p1 = np.array(nodes[bar_indices[i, 0]])
        p2 = np.array(nodes[bar_indices[i, 1]])
        l0 = np.linalg.norm(p1 - p2)
        rest_lengths[i] = l0
        
        if color == '#000000': # Black -> Rigid
            stiffness[i] = -1.0 # Flag for PBD
            damping[i] = 0.0
        else:
            # Parse Gray: #RRGGBB
            try:
                val = int(color[1:3], 16) # Extract RR
                # Map 0-255 to Stiffness range
                # Reduced max stiffness for stability
                k = 5000.0 * (1.0 - val / 255.0) + 100.0
                stiffness[i] = k
                
                # Constant damping for stability
                damping[i] = 0.5
            except:
                stiffness[i] = 1000.0 # Default
                damping[i] = 0.5

    # 5. Build Experiment Dictionary
    experiment_data = {
        'nodes': {
            'positions': np.array(nodes),
            'mass': mass,
            'attributes': attributes
        },
        'elements': {
            'bars': {
                'indices': bar_indices,
                'stiffness': stiffness,
                'rest_length': rest_lengths,
                'damping': damping
            }
        },
        'actuators': actuators,
        'meta': {
            'source': svg_path,
            'scale': scale
        }
    }
    
    return experiment_data

def save_experiment(data, output_dir):
    """
    Saves the parsed experiment data to the standard HDF5/JSON format.
    """
    import h5py
    import json
    
    out_path = Path(output_dir)
    input_dir = out_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Geometry (H5)
    with h5py.File(input_dir / "geometry.h5", 'w') as f:
        # Nodes
        g_nodes = f.create_group('nodes')
        g_nodes.create_dataset('positions', data=data['nodes']['positions'])
        g_nodes.create_dataset('masses', data=data['nodes']['mass'])
        g_nodes.create_dataset('attributes', data=data['nodes']['attributes'])
        
        # Elements
        g_elems = f.create_group('elements')
        g_bars = g_elems.create_group('bars')
        g_bars.create_dataset('indices', data=data['elements']['bars']['indices'])
        g_bars.create_dataset('stiffness', data=data['elements']['bars']['stiffness'])
        g_bars.create_dataset('rest_length', data=data['elements']['bars']['rest_length'])
        g_bars.create_dataset('damping', data=data['elements']['bars']['damping'])

    # Save Signals (H5)
    # Reduced timestep for stability
    dt = 0.001 
    duration = 5.0
    steps = int(duration / dt)
    time = np.linspace(0, duration, steps)
    
    with h5py.File(input_dir / "signals.h5", 'w') as f:
        f.attrs['dt_base'] = dt
        
        for act in data['actuators']:
            node_idx = act['node_idx']
            sig_name = act['signal_name']
            
            p0 = data['nodes']['positions'][node_idx]
            
            # Slower oscillation
            signal = np.zeros((steps, 3))
            signal[:, 0] = p0[0]
            signal[:, 1] = p0[1]
            signal[:, 2] = p0[2] + 0.5 * np.sin(2 * np.pi * 0.5 * time)
            
            f.create_dataset(sig_name, data=signal)

    # Save Config/Actuation (JSON)
    config = {
        'actuators': data['actuators'],
        'meta': data['meta'],
        'physics': {
            'gravity': -9.81,
            'global_damping': 0.1,
            'dt': dt,
            'substeps': 10
        },
        'simulation': {
            'duration': duration,
            'save_interval': 0.01,
            'dt_base': dt # Explicitly add dt_base for engine
        }
    }
    
    with open(input_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Experiment saved to {out_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python svg_importer.py <input.svg> <output_dir>")
    else:
        data = parse_svg_experiment(sys.argv[1], sys.argv[2])
        save_experiment(data, sys.argv[2])
