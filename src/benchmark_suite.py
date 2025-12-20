import time
import subprocess
import sys
import logging
import os
from pathlib import Path
from datetime import datetime

# Try to import pynvml for GPU temp monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

# Configure simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [BENCHMARK] - %(message)s')
log = logging.getLogger(__name__)

def wait_for_gpu_cooldown(target_temp: int = 65, poll_interval: int = 2):
    """
    Blocks execution indefinitely until GPU temperature drops below target_temp.
    """
    if not PYNVML_AVAILABLE:
        log.warning("pynvml not installed. Skipping smart cooldown (sleeping 10s instead).")
        time.sleep(10)
        return

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Default to GPU 0
        
        log.info(f"Waiting for GPU to cool down to {target_temp}°C (No Timeout)...")
        
        while True:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            if temp <= target_temp:
                log.info(f"Cooldown complete. Current Temp: {temp}°C")
                return
            
            # Print status every 5 seconds roughly
            if int(time.time()) % 5 == 0:
                print(f"   ... cooling ... {temp}°C", end="\r", flush=True)
                
            time.sleep(poll_interval)
            
    except Exception as e:
        log.error(f"Error reading GPU stats: {e}. Skipping cooldown.")
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass

def run_benchmark_suite(output_base: str, benchmark_configs: list, python_bin: str = sys.executable):
    """
    Orchestrates the back-to-back benchmarking.
    """
    
    output_base = Path(output_base).resolve()
    
    log.info("=========================================")
    log.info("   GSIP BENCHMARK SUITE STARTED")
    log.info("=========================================")
    log.info(f"Output Base: {output_base}")
    log.info(f"Benchmarks to Run: {len(benchmark_configs)}")
    
    results = []

    for config in benchmark_configs:
        model_name = config["name"]
        input_path = Path(config["input_path"]).resolve() # Resolve input path here
        config_overrides = config.get("config_overrides", [])
        run_label = config.get("label", model_name) # Use label for output folder if provided

        # Check if input_path exists before proceeding
        if not input_path.exists():
            log.error(f"Input path not found for {run_label}: {input_path}. Skipping benchmark.")
            results.append({"model": run_label, "status": "SKIPPED", "reason": "Input path not found"})
            continue

        log.info("")
        log.info(f"--- Preparing: {run_label} ({model_name}) ---")
        log.info(f"    Input: {input_path}")
        log.info(f"    Overrides: {config_overrides}")
        
        # 1. SMART COOLDOWN
        wait_for_gpu_cooldown(target_temp=65) # Target temp 65C
        
        # 2. DEFINE OUTPUT
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output = output_base / run_label / timestamp
        
        # 3. BUILD COMMAND
        cmd = [
            python_bin,
            "src/main.py",
            f"input_path={input_path}",
            f"output_path={run_output}",
            f"model={model_name}",
            # Force enable benchmarking flags just in case
            "pipeline.output.save_preview=true",
        ] + config_overrides # Append config overrides
        
        log.info(f"Executing: {' '.join(cmd)}")
        
        start_t = time.time()
        try:
            # Run and stream output
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=os.getcwd() # Run from project root
            )
            
            # Print output in real-time
            for line in process.stdout:
                print(f"[{model_name}] {line}", end="")
            
            process.wait() # Wait for the process to complete
            
            if process.returncode == 0:
                duration = time.time() - start_t
                log.info(f"SUCCESS: {run_label} finished in {duration:.2f}s")
                results.append({"model": run_label, "status": "OK", "duration": duration, "path": str(run_output)})
            else:
                log.error(f"FAILURE: {run_label} failed with code {process.returncode}")
                results.append({"model": run_label, "status": "FAIL", "code": process.returncode})
                
        except KeyboardInterrupt:
            log.error("Suite interrupted by user.")
            break
        except Exception as e:
            log.error(f"System error executing {model_name}: {e}")

    log.info("")
    log.info("=========================================")
    log.info("   BENCHMARK SUMMARY")
    log.info("=========================================")
    for res in results:
        print(f"{res['model']:<30} | {res['status']:<5} | {res.get('duration', 0):.2f}s")

if __name__ == "__main__":
    # CONFIGURATION
    
    # 1. Define placeholder paths for different input types
    S2_SINGLE_TEMPORAL_PATH = Path("/home/rati/extra/inputs/hamburg/S2B_MSIL2A_20250612T102559_N0511_R108_T32UNE_20250612T131505.SAFE")
    S1_S2_GROUP_PATH = Path("/home/rati/extra/inputs/hamburg")
    S2_MULTI_TEMPORAL_PATH = Path("/home/rati/extra/inputs/hamburg_series") # Example: /home/rati/extra/inputs/hamburg_series

    # 2. Define Output Location
    OUTPUT_DIR = "out/benchmarks_final"
    
    # 3. Define Models to Test with specific input paths and config overrides
    BENCHMARK_CONFIGS = [
        {"name": "resnet_s2", "input_path": S2_SINGLE_TEMPORAL_PATH, "config_overrides": []},
        {"name": "resnet_all", "input_path": S1_S2_GROUP_PATH, "config_overrides": ["data_source.use_sentinel_1=True"]},
        {"name": "convnext_s2", "input_path": S2_MULTI_TEMPORAL_PATH, "config_overrides": []},
        {"name": "prithvi_crop", "input_path": S2_MULTI_TEMPORAL_PATH, "config_overrides": []},
        {"name": "prithvi_flood_segmentation", "input_path": S2_MULTI_TEMPORAL_PATH, "config_overrides": []},
        # Add a specific ResNet model for naive tiling test
        {"name": "resnet_s2", "input_path": S2_SINGLE_TEMPORAL_PATH, "label": "resnet_s2_naive_tiling", "config_overrides": ["pipeline.tiling.overlap_ratio=0.0", "pipeline.tiling.patch_size=256", "pipeline.tiling.patch_stride=256"]},
    ]
    
    run_benchmark_suite(OUTPUT_DIR, BENCHMARK_CONFIGS) # input_path removed from run_benchmark_suite signature
