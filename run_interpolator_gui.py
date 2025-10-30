import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import threading
import random
import os
from collections import Counter

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torchvision import transforms

# --- Core Logic Imports from our Project ---
from src.config import *
from src.dataset import get_data_loader
from src.vq_vae.model import VQVAE # Use the correct VQVAE class

# --- Path to the pre-computed artifact ---
LATENT_ARTIFACT_PATH = "dataset_latents.pt"

# --- Helper Functions (All metrics and methods retained) ---

def find_nearest_neighbor_frames(model, img1_tensor, img2_tensor, dataset_latents, 
                                 full_dataset, num_steps, device, app_instance, 
                                 method='nearest_neighbor'):
    """
    Perform interpolation in VQ-VAE latent space with multiple methods and detailed metrics.
    (This function is identical to your previous version)
    """
    model.eval()
    stats_log, visual_frames = [], []
    nearest_indices = []
    
    with torch.no_grad():
        # Encode images to continuous and then quantized latent space
        z1 = model.encoder(img1_tensor.unsqueeze(0).to(device))
        quantized1, _, indices1, _ = model.vq(z1)
        
        z2 = model.encoder(img2_tensor.unsqueeze(0).to(device))
        quantized2, _, indices2, _ = model.vq(z2)
        
        print(f"\n🔍 Debug Info (Method: {method}):")
        print(f"Distance between start and end in quantized space: {torch.norm(quantized1 - quantized2).item():.4f}")
        
        for i, alpha in enumerate(np.linspace(0, 1, num_steps)):
            if app_instance:
                app_instance.update_status(f"Interpolating ({method}): Step {i+1}/{num_steps}")
            
            # Initialize metrics for the current step
            dist_to_neighbor, interp_perplexity = 0.0, 0.0

            if alpha == 0.0:
                visual_frames.append(img1_tensor)
                quantized_interp, interp_indices = quantized1, indices1
                nearest_indices.append(-1) # Marker for start
            elif alpha == 1.0:
                visual_frames.append(img2_tensor)
                quantized_interp, interp_indices = quantized2, indices2
                nearest_indices.append(-2) # Marker for end
            else:
                if method == 'direct_decode':
                    z_interp = (1 - alpha) * z1 + alpha * z2
                    quantized_interp, _, interp_indices, interp_perplexity = model.vq(z_interp)
                    decoded = model.decoder(quantized_interp)
                    visual_frames.append(torch.clamp(decoded.squeeze(0).cpu(), 0, 1))
                    nearest_indices.append(-3) # Marker for generated

                elif method in ['nearest_neighbor', 'top_k_random']:
                    z_interp_raw = (1 - alpha) * quantized1 + alpha * quantized2
                    quantized_interp, _, interp_indices, interp_perplexity = model.vq(z_interp_raw)
                    
                    # Move dataset latents to device for comparison
                    distances = torch.norm((dataset_latents.to(device) - quantized_interp).flatten(start_dim=1), dim=1)
                    
                    if method == 'nearest_neighbor':
                        nearest_idx = torch.argmin(distances).item()
                        dist_to_neighbor = distances[nearest_idx].item()
                    else: # top_k_random
                        k = min(10, len(distances))
                        top_k = torch.topk(distances, k=k, largest=False)
                        weights = 1.0 / (top_k.values + 1e-6)
                        chosen_k_idx = torch.multinomial(weights / weights.sum(), 1).item()
                        nearest_idx = top_k.indices[chosen_k_idx].item()
                        dist_to_neighbor = distances[nearest_idx].item()
                    
                    visual_frames.append(full_dataset[nearest_idx])
                    nearest_indices.append(nearest_idx)
                else:
                    raise ValueError(f"Unknown method: {method}")

            # Compute all statistics for the log
            dist_to_start = torch.norm(quantized_interp - quantized1).item()
            dist_to_end = torch.norm(quantized_interp - quantized2).item()
            unique_codes = len(torch.unique(interp_indices))
            
            # Calculate quantization error
            if alpha == 0.0 or alpha == 1.0:
                quantization_error = 0.0
            else:
                pre_quant_vec = (1 - alpha) * z1 + alpha * z2 if method == 'direct_decode' else (1 - alpha) * quantized1 + alpha * quantized2
                quantization_error = torch.norm(quantized_interp - pre_quant_vec).item()
            
            stats_log.append({
                'alpha': alpha,
                'dist_to_start': dist_to_start,
                'dist_to_end': dist_to_end,
                'unique_codes': unique_codes,
                'quantization_error': quantization_error,
                'perplexity': interp_perplexity.item() if torch.is_tensor(interp_perplexity) else interp_perplexity,
                'dist_to_neighbor': dist_to_neighbor
            })

    if method != 'direct_decode':
        unique_nearest = len(set([idx for idx in nearest_indices if idx >= 0]))
        print(f"\n📊 Path Analysis: Unique dataset emojis used: {unique_nearest} / {num_steps - 2}")
    return visual_frames, stats_log


def create_final_outputs(start_img, end_img, visual_frames, stats, filename_prefix="gui_walk", app_instance=None):
    """
    Create both the animated GIF with plots and the static summary image.
    (This function is identical to your previous version)
    """
    if app_instance: app_instance.update_status("Creating final outputs...")
    
    # --- Create the main animation with plots ---
    fig = plt.figure(figsize=(14, 12), constrained_layout=True)
    gs = fig.add_gridspec(4, 2)
    
    # Top row: Images
    ax_path = fig.add_subplot(gs[0, :])
    ax_path.set_title("Emoji Interpolation Path", fontsize=16, fontweight='bold')
    ax_path.axis('off')
    
    # --- Metric Plots ---
    ax_dist = fig.add_subplot(gs[1, :]);
    ax_neighbor = fig.add_subplot(gs[2, 0]);
    ax_codes = fig.add_subplot(gs[2, 1]);
    ax_quant = fig.add_subplot(gs[3, 0]);
    ax_perp = fig.add_subplot(gs[3, 1]);
    
    # Plot 1: Distances to endpoints
    line_start, = ax_dist.plot([], [], '#1f77b4', label='Dist to Start')
    line_end, = ax_dist.plot([], [], '#d62728', label='Dist to End')
    ax_dist.set_xlim(0, 1); ax_dist.set_ylim(0, max(s['dist_to_start'] for s in stats) * 1.1)
    ax_dist.set_title("Distance to Endpoints"); ax_dist.legend(); ax_dist.grid(True, alpha=0.5)

    # Plot 2: Distance to Manifold
    line_neighbor, = ax_neighbor.plot([], [], '#2ca02c', label='Dist to Nearest Neighbor')
    ax_neighbor.set_xlim(0, 1); ax_neighbor.set_ylim(0, max(s['dist_to_neighbor'] for s in stats) * 1.1)
    ax_neighbor.set_title("Distance to Data Manifold"); ax_neighbor.legend(); ax_neighbor.grid(True, alpha=0.5)
    
    # Plot 3: Unique Codes
    line_codes, = ax_codes.plot([], [], '#ff7f0e', label='Unique Codes')
    ax_codes.set_xlim(0, 1); ax_codes.set_ylim(0, max(s['unique_codes'] for s in stats) * 1.1)
    ax_codes.set_title("Codebook Usage"); ax_codes.legend(); ax_codes.grid(True, alpha=0.5)

    # Plot 4: Quantization Error
    line_quant, = ax_quant.plot([], [], '#9467bd', label='Quantization Error')
    ax_quant.set_xlim(0, 1); ax_quant.set_ylim(0, max(s['quantization_error'] for s in stats) * 1.1)
    ax_quant.set_title("Off-Codebook Distance"); ax_quant.legend(); ax_quant.grid(True, alpha=0.5)

    # Plot 5: Perplexity
    line_perp, = ax_perp.plot([], [], '#8c564b', label='Perplexity')
    ax_perp.set_xlim(0, 1); ax_perp.set_ylim(0, max(s['perplexity'] for s in stats) * 1.1)
    ax_perp.set_title("Codebook Perplexity"); ax_perp.legend(); ax_perp.grid(True, alpha=0.5)

    # --- Animation Update Function ---
    path_artists = []
    def update(frame_idx):
        for artist in path_artists: artist.remove()
        path_artists.clear()

        # Sliding window for emoji path
        window_size = 11
        start = max(0, frame_idx - window_size // 2)
        end = min(len(visual_frames), start + window_size)
        start = max(0, end - window_size)
        
        visible_frames = visual_frames[start:end]
        x_positions = np.linspace(0, 1, len(visible_frames))
        
        for i, frame_tensor in enumerate(visible_frames):
            img = frame_tensor.permute(1, 2, 0).numpy()
            im_ax = ax_path.inset_axes([x_positions[i] - 0.04, 0, 0.08, 1.0])
            im_ax.imshow(img)
            im_ax.axis('off')
            if start + i == frame_idx:
                 im_ax.add_patch(plt.Rectangle((0,0), 1, 1, fc='none', ec='orange', lw=4, transform=im_ax.transAxes))
            path_artists.append(im_ax)
        
        # Update metric plots
        x_data = [s['alpha'] for s in stats[:frame_idx+1]]
        line_start.set_data(x_data, [s['dist_to_start'] for s in stats[:frame_idx+1]])
        line_end.set_data(x_data, [s['dist_to_end'] for s in stats[:frame_idx+1]])
        line_neighbor.set_data(x_data, [s['dist_to_neighbor'] for s in stats[:frame_idx+1]])
        line_codes.set_data(x_data, [s['unique_codes'] for s in stats[:frame_idx+1]])
        line_quant.set_data(x_data, [s['quantization_error'] for s in stats[:frame_idx+1]])
        line_perp.set_data(x_data, [s['perplexity'] for s in stats[:frame_idx+1]])
        
        return [line_start, line_end, line_neighbor, line_codes, line_quant, line_perp] + path_artists

    anim = FuncAnimation(fig, update, frames=len(visual_frames), blit=False, interval=100)
    anim.save(f"{filename_prefix}.gif", writer='pillow', fps=10)
    plt.close(fig)
    print(f"✅ Animation saved to {filename_prefix}.gif")

# --- Main GUI Application Class ---
class InterpolatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎨 VQ-VAE Emoji Interpolator (Pre-loaded)")
        
        # --- State & Config ---
        self.selection_state = 'start'
        self.start_index = None
        self.end_index = None
        self.sample_tensors = []
        self.emoji_labels = []
        self.photo_images = []
        self.interpolation_method = tk.StringVar(value='nearest_neighbor')
        self.num_steps_var = tk.IntVar(value=64)
        
        # --- Top Control Panel ---
        top_frame = tk.Frame(root, pady=5)
        top_frame.pack(fill=tk.X, padx=10)

        method_menu = ttk.Combobox(top_frame, textvariable=self.interpolation_method, values=['nearest_neighbor', 'direct_decode', 'top_k_random'], state="readonly")
        method_menu.pack(side=tk.LEFT, padx=5)
        
        tk.Label(top_frame, text="Steps:").pack(side=tk.LEFT)
        tk.Spinbox(top_frame, from_=10, to=200, textvariable=self.num_steps_var, width=5).pack(side=tk.LEFT, padx=5)

        self.btn_regenerate_grid = tk.Button(top_frame, text="🎲 New Grid", command=self.setup_emoji_grid, state=tk.DISABLED)
        self.btn_regenerate_grid.pack(side=tk.RIGHT, padx=5)

        # --- Info Label ---
        self.info_label = tk.Label(root, text="Click an emoji to select START", font=("Helvetica", 12, "bold"), fg="green")
        self.info_label.pack(pady=5)
        
        # --- Emoji Grid with Scrollbar ---
        grid_container = tk.Frame(root)
        grid_container.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(grid_container)
        scrollbar = tk.Scrollbar(grid_container, orient="vertical", command=canvas.yview)
        self.grid_frame = tk.Frame(canvas)
        self.grid_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # --- Bottom Action Buttons ---
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)
        self.btn_reset = tk.Button(button_frame, text="🔄 Reset", command=self.reset_selection, state=tk.DISABLED)
        self.btn_reset.pack(side=tk.LEFT, padx=10)
        self.btn_generate = tk.Button(button_frame, text="✨ Generate Morph", command=self.run_interpolation_thread, state=tk.DISABLED, font=("Helvetica", 12, "bold"))
        self.btn_generate.pack(side=tk.LEFT, padx=10)
        
        # --- Status Bar ---
        self.status_label = tk.Label(root, text="Status: Initializing...", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        self.progress = ttk.Progressbar(self.status_label, mode='indeterminate')
        
        # --- Initialization ---
        self.device = torch.device(DEVICE)
        self.dataset_latents = None # Will be loaded
        self.root.after(100, self.initialize_app)

    def initialize_app(self):
        self.progress.pack(side=tk.RIGHT)
        self.progress.start()
        threading.Thread(target=self._initialize_app_thread, daemon=True).start()

    def _initialize_app_thread(self):
        try:
            if not os.path.exists(VQ_VAE_BEST_MODEL_PATH):
                raise FileNotFoundError(f"Model file not found at: {VQ_VAE_BEST_MODEL_PATH}")

            self.update_status("Loading VQ-VAE model...")
            self.model = VQVAE(
                in_channels=3, hidden_dims=VQ_VAE_HIDDEN_DIMS, latent_dim=VQ_VAE_LATENT_DIM,
                num_embeddings=VQ_VAE_NUM_EMBEDDINGS, commitment_cost=VQ_VAE_COMMITMENT_COST,
                num_res_blocks=VQ_VAE_NUM_RES_BLOCKS, ema_decay=EMA_DECAY
            ).to(self.device)
            self.model.load_state_dict(torch.load(VQ_VAE_BEST_MODEL_PATH, map_location=self.device))
            self.model.eval()
            
            self.update_status("Loading emoji dataset...")
            self.dataloader = get_data_loader()
            self.dataset = self.dataloader.dataset
            
            # --- MODIFIED BLOCK: Load pre-computed latents ---
            self.update_status("Loading pre-computed latents...")
            if not os.path.exists(LATENT_ARTIFACT_PATH):
                self.root.after(0, lambda: messagebox.showerror(
                    "Latent File Not Found",
                    f"Could not find '{LATENT_ARTIFACT_PATH}'.\n\n"
                    "Please run 'precompute_latents.py' first to generate this file."
                ))
                self.update_status(f"❌ Error: {LATENT_ARTIFACT_PATH} not found.")
                return # Stop initialization
            
            # Load the pre-computed latents from file (load to CPU)
            self.dataset_latents = torch.load(LATENT_ARTIFACT_PATH, map_location='cpu')
            self.update_status(f"Loaded {len(self.dataset_latents)} latents from file.")
            # --- END MODIFIED BLOCK ---
            
            self.root.after(0, self._finish_initialization)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Initialization Error", str(e)))
            self.update_status(f"❌ Error: {e}")

    def _finish_initialization(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.setup_emoji_grid()
        self.btn_regenerate_grid.config(state=tk.NORMAL)
        self.update_status("✅ Ready! Select a start emoji.")

    def setup_emoji_grid(self):
        self.reset_selection()
        for widget in self.grid_frame.winfo_children(): widget.destroy()
        self.sample_tensors.clear(); self.emoji_labels.clear(); self.photo_images.clear()
        
        num_samples = min(25, len(self.dataset))
        sample_indices = random.sample(range(len(self.dataset)), num_samples)
        
        for i, data_idx in enumerate(sample_indices):
            tensor = self.dataset[data_idx]
            self.sample_tensors.append(tensor)
            img = transforms.ToPILImage()(tensor).resize((64, 64), Image.Resampling.NEAREST)
            photo = ImageTk.PhotoImage(img)
            self.photo_images.append(photo)
            
            label = tk.Label(self.grid_frame, image=photo, relief="ridge", borderwidth=2, cursor="hand2")
            label.grid(row=i // 5, column=i % 5, padx=5, pady=5)
            label.bind("<Button-1>", lambda e, idx=i: self.on_emoji_click(idx))
            self.emoji_labels.append(label)

    def on_emoji_click(self, index):
        if self.selection_state == 'start':
            if self.start_index is not None: self.emoji_labels[self.start_index].config(relief="ridge", borderwidth=2, bg=self.root.cget('bg'))
            self.start_index = index
            self.emoji_labels[index].config(relief="solid", borderwidth=4, bg="lightgreen")
            self.info_label.config(text="START selected ✓ Now select an END emoji", fg="blue")
            self.selection_state = 'end'
            self.btn_reset.config(state=tk.NORMAL)
        elif self.selection_state == 'end':
            if index == self.start_index: return
            if self.end_index is not None: self.emoji_labels[self.end_index].config(relief="ridge", borderwidth=2, bg=self.root.cget('bg'))
            self.end_index = index
            self.emoji_labels[index].config(relief="solid", borderwidth=4, bg="lightcoral")
            self.info_label.config(text="START & END selected! ✓ Ready to Generate!", fg="purple")
            self.btn_generate.config(state=tk.NORMAL)

    def reset_selection(self):
        if self.start_index is not None: self.emoji_labels[self.start_index].config(relief="ridge", borderwidth=2, bg=self.root.cget('bg'))
        if self.end_index is not None: self.emoji_labels[self.end_index].config(relief="ridge", borderwidth=2, bg=self.root.cget('bg'))
        self.start_index, self.end_index = None, None
        self.selection_state = 'start'
        self.btn_generate.config(state=tk.DISABLED)
        self.btn_reset.config(state=tk.DISABLED)
        self.info_label.config(text="Click an emoji to select START", fg="green")

    def run_interpolation_thread(self):
        self.btn_generate.config(state=tk.DISABLED); self.btn_reset.config(state=tk.DISABLED); self.btn_regenerate_grid.config(state=tk.DISABLED)
        self.progress.pack(side=tk.RIGHT); self.progress.start()
        threading.Thread(target=self.run_interpolation, daemon=True).start()

    def run_interpolation(self):
        try:
            start_tensor = self.sample_tensors[self.start_index]
            end_tensor = self.sample_tensors[self.end_index]
            
            visual_frames, latent_stats = find_nearest_neighbor_frames(
                self.model, start_tensor, end_tensor, self.dataset_latents, 
                self.dataset, self.num_steps_var.get(), self.device, self, self.interpolation_method.get()
            )
            create_final_outputs(start_tensor, end_tensor, visual_frames, latent_stats, app_instance=self)
            
            self.update_status("✅ Success! GIF and summary saved.")
            self.root.after(0, lambda: messagebox.showinfo("Success!", "Files saved successfully:\n\n• gui_walk.gif\n"))
        except Exception as e:
            import traceback
            print(f"Error details:\n{traceback.format_exc()}")
            self.root.after(0, lambda: messagebox.showerror("Generation Error", f"An error occurred:\n\n{e}"))
            self.update_status(f"❌ Error: {e}")
        finally:
            self.root.after(0, self._finish_interpolation)

    def _finish_interpolation(self):
        self.progress.stop(); self.progress.pack_forget()
        self.btn_generate.config(state=tk.NORMAL if self.end_index is not None else tk.DISABLED)
        self.btn_reset.config(state=tk.NORMAL); self.btn_regenerate_grid.config(state=tk.NORMAL)

    def update_status(self, msg):
        self.root.after(0, lambda: self.status_label.config(text=f"Status: {msg}"))

if __name__ == "__main__":
    root = tk.Tk()
    app = InterpolatorGUI(root)
    root.mainloop()

