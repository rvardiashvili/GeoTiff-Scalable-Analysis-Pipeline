import os
import json
import argparse
import glob

def generate_viewer(output_dir):
    """
    Generates a main index HTML viewer for the BigEarthNet v2.0 analysis results,
    aggregating stats from all found tiles.
    """
    print(f"Generating main viewer for directory: {output_dir}")
    
    # Find all subdirectories in the output directory that seem to be tile results
    tile_dirs = []
    tile_stats = []

    if os.path.exists(output_dir):
        for d in os.listdir(output_dir):
            path = os.path.join(output_dir, d)
            if os.path.isdir(path):
                # A simple check to see if it's a tile directory
                if any(f.endswith('_classmap.json') for f in os.listdir(path)):
                    tile_dirs.append(d)
                    
                    # Try to get benchmark stats
                    stats = {
                        'name': d,
                        'duration': 'N/A',
                        'cpu': 'N/A',
                        'ram': 'N/A',
                        'gpu': 'N/A',
                        'date': 'N/A'
                    }
                    
                    bench_files = sorted(glob.glob(os.path.join(path, 'benchmark_*.json')))
                    if bench_files:
                        try:
                            with open(bench_files[-1], 'r') as f:
                                bdata = json.load(f)
                                meta = bdata.get('meta', {})
                                sys_stats = bdata.get('system_stats', {})
                                
                                stats['date'] = meta.get('start', 'N/A').split('T')[0]
                                stats['duration'] = f"{meta.get('duration_seconds', 0):.2f} s"
                                
                                if 'cpu_percent' in sys_stats:
                                    stats['cpu'] = f"{sys_stats['cpu_percent'].get('mean', 0):.1f}%"
                                if 'ram_used_gb' in sys_stats:
                                    stats['ram'] = f"{sys_stats['ram_used_gb'].get('max', 0):.2f} GB"
                                if 'gpu_util_percent' in sys_stats:
                                    stats['gpu'] = f"{sys_stats['gpu_util_percent'].get('mean', 0):.1f}%"
                        except Exception as e:
                            print(f"Error reading stats for {d}: {e}")
                    
                    tile_stats.append(stats)

                    # Generate/Update the single node viewer for this tile
                    try:
                        generate_single_node_viewer(d, output_dir)
                    except Exception as e:
                        print(f"Error generating single viewer for tile {d}: {e}")

    print(f"Found {len(tile_dirs)} tile directories.")

    # Sort tiles by name
    tile_stats.sort(key=lambda x: x['name'])

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BigEarthNet v2.0 Analysis Dashboard</title>
    <style>
        :root {{
            --primary-color: #0056b3;
            --bg-color: #f4f6f9;
            --card-bg: #ffffff;
            --text-color: #333;
            --border-radius: 8px;
            --shadow: 0 4px 6px rgba(0,0,0,0.05);
        }}
        body {{ 
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
            margin: 0; padding: 20px; 
            background-color: var(--bg-color); 
            color: var(--text-color); 
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        
        header {{ margin-bottom: 30px; border-bottom: 2px solid #e0e0e0; padding-bottom: 20px; }}
        h1 {{ margin: 0; color: var(--primary-color); }}
        
        .card {{
            background: var(--card-bg);
            border-radius: var(--border-radius);
            box-shadow: var(--shadow);
            padding: 20px;
            overflow-x: auto;
        }}
        
        table {{ width: 100%; border-collapse: collapse; min-width: 800px; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background-color: #f8f9fa; font-weight: 600; color: #444; }}
        tr:hover {{ background-color: #f9f9f9; }}
        
        a {{ color: var(--primary-color); text-decoration: none; font-weight: 500; }}
        a:hover {{ text-decoration: underline; }}
        
        .status-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            background-color: #e8f5e9;
            color: #2e7d32;
            font-size: 0.85em;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Analysis Dashboard</h1>
            <p style="color: #777;">Overview of processed BigEarthNet v2.0 tiles.</p>
        </header>

        <div class="card">
            <table>
                <thead>
                    <tr>
                        <th>Tile Name</th>
                        <th>Date</th>
                        <th>Duration</th>
                        <th>Mean CPU</th>
                        <th>Max RAM</th>
                        <th>GPU Util</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
    """

    for t in tile_stats:
        html_content += f"""
                    <tr>
                        <td><a href="{t['name']}/viewer.html">{t['name']}</a></td>
                        <td>{t['date']}</td>
                        <td>{t['duration']}</td>
                        <td>{t['cpu']}</td>
                        <td>{t['ram']}</td>
                        <td>{t['gpu']}</td>
                        <td><span class="status-badge">Analyzed</span></td>
                    </tr>
        """

    if not tile_stats:
        html_content += """
                    <tr>
                        <td colspan="7" style="text-align: center; color: #777;">No analyzed tiles found in this directory.</td>
                    </tr>
        """

    html_content += """
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""

    viewer_path = os.path.join(output_dir, 'viewer.html')
    with open(viewer_path, 'w') as f:
        f.write(html_content)
    print(f"Viewer generated at: {os.path.abspath(viewer_path)}")


def generate_single_node_viewer(tile_name, output_dir):
    """
    Generates an HTML viewer for a single BigEarthNet v2.0 analysis result.
    """
    print(f"Generating single node viewer for tile: {tile_name}")
    tile_dir = os.path.join(output_dir, tile_name)
    
    # 1. Load Benchmark Data
    bench_files = sorted(glob.glob(os.path.join(tile_dir, 'benchmark_*.json')))
    bench_data = {}
    if bench_files:
        try:
            with open(bench_files[-1], 'r') as f:
                bench_data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load benchmark data: {e}")

    # 2. Load Global Probs
    probs_file = os.path.join(tile_dir, f"{tile_name}_global_probs.json")
    global_probs = []
    if os.path.exists(probs_file):
        try:
            with open(probs_file, 'r') as f:
                d = json.load(f)
                global_probs = d.get('global_probs', [])
        except Exception as e:
            print(f"Warning: Could not load global probs: {e}")
            
    # 3. Load Classmap (for legends and labels)
    classmap_file = os.path.join(tile_dir, f"{tile_name}_classmap.json")
    class_map = {}
    if os.path.exists(classmap_file):
        try:
            with open(classmap_file, 'r') as f:
                class_map = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load classmap: {e}")

    # Determine Labels for Probs
    # Strategy 1: Use labels from classmap (Dynamic/Model-Agnostic)
    labels = []
    if class_map:
        # Sort by index if possible, otherwise just take keys
        # class_map structure: {"Label": {"index": i, "color": ...}, ...}
        try:
            sorted_items = sorted(
                [k for k in class_map.keys() if k != 'No_Dominant_Class'], 
                key=lambda k: class_map[k].get('index', 0)
            )
            labels = sorted_items
        except:
             labels = [k for k in class_map.keys() if k != 'No_Dominant_Class']

    # Strategy 2: Fallback to Standard BigEarthNet Labels
    if not labels or (global_probs and len(labels) != len(global_probs)):
        ben_labels = [
            'Agro-forestry areas', 'Arable land', 'Beaches, dunes, sands', 
            'Broad-leaved forest', 'Coastal wetlands', 'Complex cultivation patterns', 
            'Coniferous forest', 'Industrial or commercial units', 'Inland waters', 
            'Inland wetlands', 'Land principally occupied by agriculture, with significant areas of natural vegetation', 
            'Marine waters', 'Mixed forest', 'Moors, heathland and sclerophyllous vegetation', 
            'Natural grassland and sparsely vegetated areas', 'Pastures', 'Permanent crops', 
            'Transitional woodland, shrub', 'Urban fabric'
        ]
        if global_probs and len(global_probs) == len(ben_labels):
            labels = ben_labels

    # Combine probs with labels
    probs_display = []
    if global_probs and labels and len(global_probs) == len(labels):
        for lbl, p in zip(labels, global_probs):
            probs_display.append({'label': lbl, 'prob': p})
        probs_display.sort(key=lambda x: x['prob'], reverse=True)
    elif global_probs:
         # Fallback if length mismatch or no labels found
         for i, p in enumerate(global_probs):
            probs_display.append({'label': f'Class {i}', 'prob': p})
         probs_display.sort(key=lambda x: x['prob'], reverse=True)

    # 4. Prepare Benchmark Stats for Template
    stats = {
        'duration': 'N/A', 'cpu_mean': 'N/A', 'cpu_max': 'N/A',
        'ram_mean': 'N/A', 'ram_max': 'N/A', 'ram_used': 'N/A',
        'gpu_util': 'N/A', 'gpu_mem': 'N/A'
    }
    if bench_data:
        meta = bench_data.get('meta', {})
        sys_stats = bench_data.get('system_stats', {})
        
        stats['duration'] = f"{meta.get('duration_seconds', 0):.2f} s"
        
        if 'cpu_percent' in sys_stats:
            stats['cpu_mean'] = f"{sys_stats['cpu_percent'].get('mean', 0):.1f}%"
            stats['cpu_max'] = f"{sys_stats['cpu_percent'].get('max', 0):.1f}%"
            
        if 'ram_percent' in sys_stats:
            stats['ram_mean'] = f"{sys_stats['ram_percent'].get('mean', 0):.1f}%"
            stats['ram_max'] = f"{sys_stats['ram_percent'].get('max', 0):.1f}%"
            
        if 'ram_used_gb' in sys_stats:
             stats['ram_used'] = f"{sys_stats['ram_used_gb'].get('max', 0):.2f} GB"

        if 'gpu_util_percent' in sys_stats:
             stats['gpu_util'] = f"{sys_stats['gpu_util_percent'].get('mean', 0):.1f}%"
             
        if 'gpu_mem_used_gb' in sys_stats:
             stats['gpu_mem'] = f"{sys_stats['gpu_mem_used_gb'].get('max', 0):.2f} GB"


    # 5. Build HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis Report: {tile_name}</title>
    <style>
        :root {{
            --primary-color: #0056b3;
            --bg-color: #f4f6f9;
            --card-bg: #ffffff;
            --text-color: #333;
            --border-radius: 8px;
            --shadow: 0 4px 6px rgba(0,0,0,0.05);
        }}
        body {{ 
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
            margin: 0; padding: 20px; 
            background-color: var(--bg-color); 
            color: var(--text-color); 
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        
        header {{ margin-bottom: 30px; border-bottom: 2px solid #e0e0e0; padding-bottom: 20px; }}
        h1 {{ margin: 0; color: var(--primary-color); }}
        h2 {{ color: #444; margin-top: 0; font-size: 1.4em; border-left: 5px solid var(--primary-color); padding-left: 10px; }}
        
        .section {{ margin-bottom: 40px; }}
        
        /* Stats Table */
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .stat-card {{ background: var(--card-bg); padding: 15px; border-radius: var(--border-radius); box-shadow: var(--shadow); }}
        .stat-label {{ font-size: 0.85em; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }}
        .stat-value {{ font-size: 1.5em; font-weight: bold; color: var(--primary-color); margin-top: 5px; }}
        
        /* Probs Bar Chart */
        .probs-container {{ background: var(--card-bg); padding: 20px; border-radius: var(--border-radius); box-shadow: var(--shadow); }}
        .prob-row {{ display: flex; align-items: center; margin-bottom: 8px; }}
        .prob-label {{ width: 250px; font-size: 0.9em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .prob-bar-container {{ flex-grow: 1; background-color: #eee; height: 12px; border-radius: 6px; overflow: hidden; margin: 0 15px; }}
        .prob-bar {{ height: 100%; background-color: var(--primary-color); }}
        .prob-value {{ width: 60px; text-align: right; font-size: 0.85em; font-weight: bold; color: #555; }}

        /* Preview Maps */
        .maps-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 25px; }}
        .map-card {{ background: var(--card-bg); border-radius: var(--border-radius); box-shadow: var(--shadow); overflow: hidden; display: flex; flex-direction: column; }}
        .map-header {{ padding: 15px; background-color: #f8f9fa; border-bottom: 1px solid #eee; font-weight: 600; }}
        .map-body {{ padding: 15px; text-align: center; background-color: #fafafa; }}
        .map-body img {{ max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .map-legend {{ padding: 15px; background-color: #fff; border-top: 1px solid #eee; font-size: 0.9em; }}
        
        /* Categorical Legend */
        .legend-cat {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 8px; }}
        .legend-item {{ display: flex; align-items: center; }}
        .legend-color {{ width: 12px; height: 12px; border-radius: 2px; margin-right: 8px; border: 1px solid rgba(0,0,0,0.1); }}
        .legend-text {{ font-size: 0.8em; color: #555; }}

        /* Continuous Legend */
        .colorbar {{ height: 15px; width: 100%; margin-bottom: 5px; border-radius: 3px; }}
        .colorbar-labels {{ display: flex; justify-content: space-between; font-size: 0.8em; color: #666; }}
        
        /* Gradients */
        .grad-viridis {{ background: linear-gradient(to right, #440154, #482878, #3e4989, #31688e, #26828e, #1f9e89, #35b779, #6ece58, #b5de2b, #fde725); }}
        .grad-magma {{ background: linear-gradient(to right, #000004, #140e36, #3b0f70, #641a80, #8c2981, #b73779, #de4968, #f7705c, #fe9f6d, #fcfdbf); }}
        .grad-plasma {{ background: linear-gradient(to right, #0d0887, #46039f, #7201a8, #9c179e, #bd3786, #d8576b, #ed7953, #fb9f3a, #fdca26, #f0f921); }}

    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Analysis Report</h1>
            <p style="color: #777;">Tile: <strong>{tile_name}</strong></p>
        </header>

        <!-- BENCHMARK STATS -->
        <section class="section">
            <h2>Benchmark Summary</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Duration</div>
                    <div class="stat-value">{stats['duration']}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Max RAM Used</div>
                    <div class="stat-value">{stats['ram_used']}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Mean CPU</div>
                    <div class="stat-value">{stats['cpu_mean']}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">GPU Utilization</div>
                    <div class="stat-value">{stats['gpu_util']}</div>
                </div>
            </div>
        </section>

        <!-- GLOBAL PROBABILITIES -->
        <section class="section">
            <h2>Global Class Probabilities</h2>
            <div class="probs-container">
    """
    
    if probs_display:
        # Inject Probs Rows
        for p in probs_display:
            pct = p['prob'] * 100
            html_content += f"""
                    <div class="prob-row">
                        <div class="prob-label" title="{p['label']}">{p['label']}</div>
                        <div class="prob-bar-container">
                            <div class="prob-bar" style="width: {pct:.1f}%"></div>
                        </div>
                        <div class="prob-value">{pct:.1f}%</div>
                    </div>
            """
    else:
        html_content += f"""
            <p style="color: #a00;">No global probability data found.</p>
            <p style="font-size: 0.8em; color: #777;">Checked path: {probs_file}</p>
        """
    
    html_content += """
            </div>
        </section>

        <!-- PREVIEW MAPS -->
        <section class="section">
            <h2>Maps & Visualization</h2>
            <div class="maps-grid">
    """
    
    # 1. Classification Map Card
    html_content += f"""
                <div class="map-card">
                    <div class="map-header">Classification Map</div>
                    <div class="map-body">
                        <img src="preview_class.png" alt="Classification Map" onerror="this.style.display='none'; this.parentElement.innerHTML='<p>Image not found</p>'">
                    </div>
                    <div class="map-legend">
                        <div class="legend-cat">
    """
    # Inject Legend Items
    if class_map:
        for lbl, meta in class_map.items():
            if lbl == 'No_Dominant_Class': continue
            c = meta.get('color_rgb', meta.get('color', [0,0,0]))
            html_content += f"""
                            <div class="legend-item">
                                <div class="legend-color" style="background-color: rgb({c[0]},{c[1]},{c[2]});"></div>
                                <div class="legend-text">{lbl.replace('_', ' ')}</div>
                            </div>
            """
    html_content += """
                        </div>
                    </div>
                </div>
    """

    # 2. Confidence Map Card (MaxProb)
    html_content += f"""
                <div class="map-card">
                    <div class="map-header">Confidence (Max Probability)</div>
                    <div class="map-body">
                        <img src="preview_maxprob.png" alt="Max Probability" onerror="this.style.display='none'; this.parentElement.innerHTML='<p>Image not found</p>'">
                    </div>
                    <div class="map-legend">
                        <div class="colorbar grad-viridis"></div>
                        <div class="colorbar-labels">
                            <span>0.0 (Low)</span>
                            <span>1.0 (High)</span>
                        </div>
                    </div>
                </div>
    """

    # 3. Entropy Map Card
    html_content += f"""
                <div class="map-card">
                    <div class="map-header">Uncertainty (Entropy)</div>
                    <div class="map-body">
                        <img src="preview_entropy.png" alt="Entropy" onerror="this.style.display='none'; this.parentElement.innerHTML='<p>Image not found</p>'">
                    </div>
                    <div class="map-legend">
                        <div class="colorbar grad-magma"></div>
                        <div class="colorbar-labels">
                            <span>0.0 (Certain)</span>
                            <span>High (Uncertain)</span>
                        </div>
                    </div>
                </div>
    """

    # 4. Gap Map Card
    html_content += f"""
                <div class="map-card">
                    <div class="map-header">Prediction Gap (Margin)</div>
                    <div class="map-body">
                        <img src="preview_gap.png" alt="Prediction Gap" onerror="this.style.display='none'; this.parentElement.innerHTML='<p>Image not found</p>'">
                    </div>
                    <div class="map-legend">
                        <div class="colorbar grad-plasma"></div>
                        <div class="colorbar-labels">
                            <span>0.0 (Small Margin)</span>
                            <span>1.0 (Large Margin)</span>
                        </div>
                    </div>
                </div>
    """

    html_content += """
            </div>
        </section>
    </div>
</body>
</html>
"""

    viewer_path = os.path.join(tile_dir, 'viewer.html')
    os.makedirs(os.path.dirname(viewer_path), exist_ok=True)
    with open(viewer_path, 'w') as f:
        f.write(html_content)
    print(f"Viewer generated at: {os.path.abspath(viewer_path)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate an HTML viewer for BigEarthNet v2.0 analysis results.')
    parser.add_argument('--output_dir', type=str, default='out', help='Path to the output directory.')
    parser.add_argument('--tile_name', type=str, help='Name of the tile for single node viewer.')
    args = parser.parse_args()
    
    if args.tile_name:
        generate_single_node_viewer(args.tile_name, args.output_dir)
    else:
        generate_viewer(args.output_dir)