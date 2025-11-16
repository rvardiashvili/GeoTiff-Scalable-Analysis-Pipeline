import os
import json
import argparse

def generate_viewer(output_dir):
    """
    Generates an HTML viewer for the BigEarthNet v2.0 analysis results.
    """
    print(f"Generating viewer for directory: {output_dir}")
    
    # Find all subdirectories in the output directory that seem to be tile results
    tile_dirs = []
    for d in os.listdir(output_dir):
        path = os.path.join(output_dir, d)
        if os.path.isdir(path):
            # A simple check to see if it's a tile directory
            if any(f.endswith('_classmap.json') for f in os.listdir(os.path.join(path,d))):
                tile_dirs.append(d)

    print(f"Found tile directories: {tile_dirs}")

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BigEarthNet v2.0 Viewer</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 2em; background-color: #f9f9f9; color: #333; }}
        h1, h2 {{ color: #111; }}
        #report table {{ border-collapse: collapse; width: 100%; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        #report th, #report td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        #report th {{ background-color: #f2f2f2; font-weight: 600; }}
        #report tr:nth-child(even) {{ background-color: #fdfdfd; }}
        #report tr:hover {{ background-color: #f1f1f1; }}
        .tile {{ background-color: #fff; border: 1px solid #ccc; border-radius: 8px; padding: 1.5em; margin-bottom: 2em; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .tile img {{ max-width: 100%; border-radius: 4px; }}
        .legend {{ columns: 2; -webkit-columns: 2; -moz-columns: 2; margin-top: 1em; }}
        .legend-item {{ display: flex; align-items: center; margin-bottom: 8px; }}
        .legend-color {{ width: 20px; height: 20px; margin-right: 10px; border: 1px solid #000; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>BigEarthNet v2.0 Analysis Viewer</h1>

    <h2>Benchmark Report</h2>
    <div id="report">Loading report...</div>

    <h2>Analyzed Tiles</h2>
    <div id="tiles"></div>

    <script>
        const tileDirs = {json.dumps(tile_dirs)};

        // Display benchmark report
        fetch('benchmark_report.csv')
            .then(response => {{
                if (!response.ok) {{
                    throw new Error('benchmark_report.csv not found.');
                }}
                return response.text();
            }})
            .then(data => {{
                const rows = data.trim().split('\\n').map(row => row.split(','));
                if (rows.length < 1) return;
                let table = '<table>';
                table += '<thead><tr>' + rows[0].map(header => `<th>${{header}}</th>`).join('') + '</tr></thead>';
                table += '<tbody>';
                for (let i = 1; i < rows.length; i++) {{
                    if (rows[i].length === rows[0].length) {{
                        table += '<tr>' + rows[i].map(cell => `<td>${{cell}}</td>`).join('') + '</tr>';
                    }}
                }}
                table += '</tbody></table>';
                document.getElementById('report').innerHTML = table;
            }})
            .catch(error => {{
                document.getElementById('report').innerHTML = `<p>${{error.message}}</p>`;
            }});

        // Display tiles
        const tilesContainer = document.getElementById('tiles');
        if (tileDirs.length === 0) {{
            tilesContainer.innerHTML = '<p>No analyzed tiles found.</p>';
        }} else {{
            tileDirs.forEach(dir => {{
                const tileElement = document.createElement('div');
                tileElement.className = 'tile';
                tileElement.innerHTML = `<h3>${{dir}}</h3>`;

                const previewImg = document.createElement('img');
                const previewPath = `${{dir}}/${{dir}}/preview.png`;
                previewImg.src = previewPath;
                previewImg.alt = `Preview for ${{dir}}`;
                previewImg.onerror = () => {{ previewImg.alt = `Preview not found at ${{previewPath}}`; previewImg.style.display='none'; }};
                tileElement.appendChild(previewImg);

                const legendElement = document.createElement('div');
                const classmapPath = `${{dir}}/${{dir}}/${{dir}}_classmap.json`;
                fetch(classmapPath)
                    .then(response => {{
                        if (!response.ok) {{
                            throw new Error(`classmap.json not found for ${{dir}}`);
                        }}
                        return response.json();
                    }})
                    .then(classmap => {{
                        let legendHtml = '<h4>Legend</h4><div class="legend">';
                        for (const label in classmap) {{
                            if (label !== 'No_Dominant_Class') {{
                                const color = classmap[label].color_rgb;
                                legendHtml += `
                                    <div class="legend-item">
                                        <div class="legend-color" style="background-color: rgb(${{color[0]}}, ${{color[1]}}, ${{color[2]}})"></div>
                                        <span>${{label.replace(/_/g, ' ')}}</span>
                                    </div>`;
                            }}
                        }}
                        legendHtml += '</div>';
                        legendElement.innerHTML = legendHtml;
                    }})
                    .catch(error => {{
                        legendElement.innerHTML = `<p>${{error.message}}</p>`;
                    }});
                tileElement.appendChild(legendElement);

                tilesContainer.appendChild(tileElement);
            }});
        }}
    </script>
</body>
</html>
"""

    viewer_path = os.path.join(output_dir, 'viewer.html')
    with open(viewer_path, 'w') as f:
        f.write(html_content)
    print(f"Viewer generated at: {os.path.abspath(viewer_path)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate an HTML viewer for BigEarthNet v2.0 analysis results.')
    parser.add_argument('--output_dir', type=str, default='out', help='Path to the output directory.')
    args = parser.parse_args()
    
    generate_viewer(args.output_dir)
