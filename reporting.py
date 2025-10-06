"""
Functions for post-processing analysis results and generating the final HTML report
without storing full patch tensors in memory.
"""
from typing import Dict, Any, List
import numpy as np
from jinja2 import Template

from config import DATA_LOADER_WORKERS, GPU_BATCH_SIZE
from utils import NEW_LABELS

def generate_html_report_data(scene_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculates scene-wide stats for the report without using patch tensors.
    """
    scene_patches = scene_results["patches"]

    # 1. Scene-wide mean probabilities
    scene_results["top_labels"] = sorted(
        [(k, np.mean([p["probs"][NEW_LABELS.index(k)] for p in scene_patches]))
         for k in NEW_LABELS],
        key=lambda x: x[1], reverse=True
    )[:5]

    # 2. Sort patches by highest probability score
    sorted_patches = sorted(scene_patches, key=lambda x: x["max_prob"], reverse=True)

    # 3. Keep only metadata needed for top 9 patches
    for patch in sorted_patches[:9]:
        # Remove any patch_tensor_cpu references if present
        patch.pop("patch_tensor_cpu", None)
        # Optionally, you can add a placeholder image
        patch["base64_img"] = ""  # empty string or a small placeholder image

    scene_results["patches"] = sorted_patches
    return scene_results

def generate_html_report(scenes_results: List[Dict], total_time: float, avg_gpu_batch_time: float,
                         total_patches: int, output_file: str = "report.html"):
    """
    Generates the final HTML report using metadata only (no tensors).
    """
    for scene in scenes_results:
        scene['total_patches'] = len(scene['patches'])

    HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>BigEarthNet Scene Report</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 20px; background-color: #f4f7f9; color: #333; }
h1 { color: #004d99; border-bottom: 2px solid #004d99; padding-bottom: 10px; }
h2 { color: #3366cc; margin-top: 30px; }
.summary-box { background-color: #e6f0ff; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid #3366cc; }
.scene-container { background-color: #fff; border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
.patch-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; }
.patch-card { border: 1px solid #ccc; padding: 10px; background-color: #f9f9f9; border-radius: 4px; text-align: center; }
.patch-card img { width: 100%; height: auto; display: block; border-radius: 3px; background-color: #ddd; }
.patch-details { font-size: 0.9em; margin-top: 5px; }
</style>
</head>
<body>
<h1>BigEarthNet Scene Analysis Report</h1>
<div class="summary-box">
    <p><strong>Total Scenes Analyzed:</strong> {{ scenes | length }}</p>
    <p><strong>Total Patches Processed:</strong> {{ total_patches }}</p>
    <p><strong>Total Processing Time:</strong> {{ "%.2f"|format(total_time) }} seconds</p>
    <p><strong>Average GPU Batch Time ({{ GPU_BATCH_SIZE }} patches):</strong> {{ "%.4f"|format(avg_gpu_batch_time) }} seconds</p>
    <p><strong>Parallel Config (Multiprocessing):</strong> {{ DATA_LOADER_WORKERS }} DataLoader Workers -> 1 GPU Consumer (batch {{ GPU_BATCH_SIZE }})</p>
</div>
{% for scene in scenes %}
<div class="scene-container">
    <h2>Scene: {{ scene.name }} ({{ scene.total_patches }} patches)</h2>
    <h3>Top 5 Scene Labels (Mean Probability)</h3>
    <ul>{% for label, prob in scene.top_labels %}<li><strong>{{ label }}:</strong> {{ "%.4f"|format(prob) }}</li>{% endfor %}</ul>
    <h3>Sample Patches (Top 9 by Max Probability)</h3>
    <div class="patch-grid">
    {% for patch in scene.patches[:9] %}
        <div class="patch-card">
            <img src="data:image/png;base64,{{ patch.base64_img }}" alt="{{ patch.name }}">
            <div class="patch-details">
                <p><strong>Patch:</strong> {{ patch.name }}</p>
                <p><strong>Top Label:</strong> {{ patch.top_label_name }} ({{ "%.4f"|format(patch.top_label_prob) }})</p>
            </div>
        </div>
    {% endfor %}
    </div>
</div>
{% endfor %}
</body>
</html>
"""
    template = Template(HTML_TEMPLATE)
    html = template.render(
        scenes=scenes_results,
        total_patches=total_patches,
        total_time=total_time,
        avg_gpu_batch_time=avg_gpu_batch_time,
        DATA_LOADER_WORKERS=DATA_LOADER_WORKERS,
        GPU_BATCH_SIZE=GPU_BATCH_SIZE,
    )
    with open(output_file, "w") as f:
        f.write(html)
    print(f"✅ HTML report saved to {output_file}")
