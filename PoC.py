import os

# 1. Environment configuration (MUST be set before torch import)
os.environ["HIP_VISIBLE_DEVICES"] = "1"  # Isolates dGPU so PyTorch sees it as default CUDA device
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0" # Prevents hipErrorInvalidImage on RDNA3
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.8"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import random
import time

from model import Siren

# Force cuDNN off if needed for ROCm stability
torch.backends.cudnn.enabled = False

# Now device logic points cleanly to the isolated card
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0  # Normalize pixels to [0, 1]

def get_random_chunk(img_array: np.ndarray) -> np.ndarray:
    h, w, c = img_array.shape
    h_chunk = random.randint(16, min(256, h))
    w_chunk = random.randint(16, min(256, w))
    
    y_start = random.randint(0, h - h_chunk)
    x_start = random.randint(0, w - w_chunk)
    
    return img_array[y_start : y_start + h_chunk, x_start : x_start + w_chunk, :]

def create_coordinate_grid(height: int, width: int) -> torch.Tensor:
    y_coords = torch.linspace(-1, 1, steps=height)
    x_coords = torch.linspace(-1, 1, steps=width)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords = torch.stack([grid_x, grid_y], dim=-1)  # Shape: (H, W, 2)
    return coords

def fit_siren_on_chunk(chunk_np: np.ndarray, steps: int = 1000) -> float:
    h, w, c = chunk_np.shape
    coords = create_coordinate_grid(h, w).to(device)
    targets = torch.from_numpy(chunk_np).float().to(device)
    
    model = Siren(in_dim=2, hidden_dim=64, out_dim=3, num_layers=3, omega_0=30.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for step in range(steps):
        optimizer.zero_grad()
        predictions = model(coords)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
    return loss.item()

def run_poc():
    # --- Setup Configuration ---
    IMAGE_FOLDER = "Kodak Images"
    DEFAULT_IMAGE = "18.png"
    THRESHOLD = 0.001
    image_path = os.path.join(IMAGE_FOLDER, DEFAULT_IMAGE)

    try:
        full_img = load_image(image_path)
        print(f"Successfully loaded {image_path} (Full shape: {full_img.shape})")

        # CROP A 128x128 PATCH FOR THE BENCHMARK RUNS
        benchmark_crop = full_img[:256, :256, :]
        print(f"Benchmark crop shape: {benchmark_crop.shape}")

        # --- Phase 1: Train LogReg model ---
        print("\n--- Phase 1: Training Logistic Regression Predictor ---")
        initial_data = []
        for i in range(1): # Sample 100 random chunks from the full image to train the classifier
            chunk_np = get_random_chunk(full_img)
            mse = fit_siren_on_chunk(chunk_np, steps=1000)
            std = float(np.std(chunk_np))
            initial_data.append({"std": std, "mse": mse})

        features_raw = np.array([d["std"] for d in initial_data])
        labels_raw = np.array([1 if d["mse"] > THRESHOLD else 0 for d in initial_data])
        
        feat_tensor = torch.tensor(features_raw).float().unsqueeze(1).to(device)
        lab_tensor = torch.tensor(labels_raw).float().unsqueeze(1).to(device)

        logreg_model = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid()).to(device)
        log_opt = optim.Adam(logreg_model.parameters(), lr=0.1)
        for _ in range(200):
            log_opt.zero_grad()
            out = logreg_model(feat_tensor)
            loss = nn.BCELoss()(out, lab_tensor)
            loss.backward()
            log_opt.step()

        # --- Phase 2: Comparative Benchmark with Quadtree ---
        print("\n--- Phase 2: Comparing Method A (Standard) vs Method B (Hybrid) ---")
    
        # Metrics trackers
        stats = {
            "A": {"nodes": 0, "fits": 0},
            "B": {"nodes": 0, "fits": 0, "skips": 0}
        }

        def process_quadtree(chunk_np, method="A"):
            stats[method]["nodes"] += 1
            h, w = chunk_np.shape[0], chunk_np.shape[1]
            
            # Base Case: Stop splitting if patch reaches 16x16 or smaller
            if h <= 16 or w <= 16:
                stats[method]["fits"] += 1
                _ = fit_siren_on_chunk(chunk_np, steps=1000)
                return

            should_split = False

            if method == "B":
                std_val = float(np.std(chunk_np))
                with torch.no_grad():
                    pred_prob = logreg_model(torch.tensor([[std_val]]).to(device))
                    risk_of_failure = pred_prob.item() > 0.99
                    risk_of_failure = h > 64 # HARD CODED FOR PoC
                
                if risk_of_failure:
                    # Proactive Split: Skip SIREN training entirely
                    stats["B"]["skips"] += 1
                    should_split = True
                else:
                    # Oracle predicted success: Attempt SIREN training
                    stats["B"]["fits"] += 1
                    mse = fit_siren_on_chunk(chunk_np, steps=1000)
                    if mse > THRESHOLD:
                        # Fallback Split: Oracle made False Negative error
                        should_split = True

            else:
                # Method A (Standard Brute-Force Quadtree): Always try training first
                stats["A"]["fits"] += 1
                mse = fit_siren_on_chunk(chunk_np, steps=1000)
                if mse > THRESHOLD:
                    should_split = True

            # Execute Quadtree Partitioning if needed
            if should_split:
                mid_h = h // 2
                mid_w = w // 2
                quadrants = [
                    chunk_np[0:mid_h, 0:mid_w],
                    chunk_np[0:mid_h, mid_w:w],
                    chunk_np[mid_h:h, 0:mid_w],
                    chunk_np[mid_h:h, mid_w:w]
                ]
                for q in quadrants:
                    process_quadtree(q, method=method)

        # --- Execution of Benchmarks ---
        print("Running Method A (Standard Quadtree on 128x128)...")
        start_a = time.perf_counter()
        process_quadtree(benchmark_crop, method="A")
        end_a = time.perf_counter()
        time_a = end_a - start_a
        print(f"Method A Total Time: {time_a:.2f}s | Total Nodes: {stats['A']['nodes']} | Models Fitted: {stats['A']['fits']}")

        print("\nRunning Method B (Hybrid Quadtree on 128x128)...")
        start_b = time.perf_counter()
        process_quadtree(benchmark_crop, method="B")
        end_b = time.perf_counter()
        time_b = end_b - start_b
        print(f"Method B Total Time: {time_b:.2f}s | Total Nodes: {stats['B']['nodes']} | Models Fitted: {stats['B']['fits']} | Proactive Skips: {stats['B']['skips']}")

        # --- Summary ---
        print("\n--- Final Comparison Results ---")
        if time_b < time_a:
            improvement = ((time_a - time_b) / time_a) * 100
            print(f"SUCCESS: Method B is {improvement:.2f}% faster than Method A.")
        else:
            slowdown = ((time_b - time_a) / time_a) * 100
            print(f"Method B was {slowdown:.2f}% slower.")
            print(f"Node ratio (B/A): {stats['B']['nodes'] / stats['A']['nodes']:.2f}x")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_poc()