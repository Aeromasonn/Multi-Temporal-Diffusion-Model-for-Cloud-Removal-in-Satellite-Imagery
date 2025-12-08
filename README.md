This is the COMPSCI-372 Final Project of Sihan Yao & Yuxuan Huang.
# **Downstream Task: DEM Generation from Cloud-Removed Sentinel-2 Imagery**

## **1. Overview**

After completing cloud removal with our multi-temporal diffusion model, the downstream task predicts a **Digital Elevation Model (DEM)** from each cloud-free RGB tile.  
This module is based on a modified version of the **ImageToDEM** pix2pix generator proposed by *Panagiotou et al., 2020* in *Remote Sensing* (MDPI) :contentReference[oaicite:0]{index=0}.

Our goal is:

> **Given a 128×128×4 multi-spectral patch, predict a 128×128 DEM tile using a pretrained U-Net generator.**

We evaluate how diffusion-based cloud removal improves DEM estimation compared to using cloudy RGB directly.

---

## **2. What Our Downstream Pipeline Does**

### **2.1 Input Handling**

Each sample is a **4-channel** Sentinel-2 patch:

- **Channels 0–2:** RGB  
- **Channel 3:** NIR  

But the pretrained ImageToDEM generator only accepts **3-channel RGB**.  
Therefore, our preprocessing steps:

1. **Drop NIR channel**
   ```python
   rgb = patch[:3]  # shape: (3, 128, 128)
   ```

2. **Normalize to [-1, 1]** (required by pix2pix)
   ```python
   rgb = (rgb / 255.0) * 2 - 1
   ```

3. **Feed into U-Net generator**  
   ```python
   pred_dem = generator(rgb[None])  # shape: (1, 1, 128, 128)
   ```

### **2.2 Generator (U-Net Architecture)**

Our downstream DEM generator is the TensorFlow pix2pix U-Net from ImageToDEM:

- **Encoder:** 8 downsampling blocks (Conv + LeakyReLU)  
- **Bottleneck:** 1×1×512  
- **Decoder:** 8 upsampling blocks with skip connections  
- **Output activation:** `tanh` → produces DEM in **[-1, 1]**

Unlike the original CGAN formulation, we use:

> **Inference only — no discriminator, no adversarial training.**

### **2.3 Output Post-Processing**

The raw output is in **[-1, 1]**:

```python
dem_pred ∈ [-1, 1]
dem_vis  = (dem_pred + 1) / 2  # → [0, 1]
```

Since the generator predicts **relative elevation structure**, not global height, we do *not* convert to meters (see Section 5).

---

## **3. Inference Workflow (Simplified)**

```python
# 1. Load pretrained pix2pix DEM generator
model = tf.keras.models.load_model(MODEL_PATH)

# 2. Prepare RGB input
rgb = clean_or_pred[:3]          # drop NIR
rgb = (rgb / 255.0) * 2 - 1      # normalization

# 3. Generate DEM
pred = model(rgb[None])

# 4. Convert for visualization
pred_vis = (pred + 1) / 2
```

We run this pipeline on:

- **VAL set:** 2048 samples  
- **TEST set:** 2053 samples  

---

## **4. Evaluation Results (Actual Console Output)**

### **Validation Set**
| Comparison | MAE |
|-----------|------|
| Cloudy RGB → DEM (baseline) | **0.215648** |
| Clean RGB (ours) → DEM | **0.071100** |

### **Test Set**
| Comparison | MAE |
|-----------|------|
| Cloudy RGB → DEM (baseline) | **0.217131** |
| Clean RGB (ours) → DEM | **0.074932** |

### **Interpretation**

Clouds significantly degrade DEM estimation accuracy.  
Our diffusion-cleaned inputs reduce MAE by **≈ 3×**, demonstrating that:

> Cloud removal improves downstream geophysical inference, not just visual quality.

---

## **5. Height Interpretation — Relative vs. Absolute DEMs**

The *Remote Sensing* paper clarifies (Sections 4–5) that:

### **5.1 DEMs must be normalized to [-1, 1]**
> “Normalisation is applied and results are expected to be in the same range… predicting the exact height map… is extremely challenging without global context.”  
— *Panagiotou et al., 2020* :contentReference[oaicite:1]{index=1}

Thus, the model learns **shape**, not absolute altitude.

### **5.2 Why absolute height is impossible in patch-based learning**

- A 128×128 patch does **not contain global elevation reference** (min/max height of the scene).  
- Neighboring tiles can have large offset differences (e.g., mountains vs. valleys).  
- Without global metadata, the generator can only predict **relative elevations**.

Therefore, our DEM outputs reflect:

- Terrain geometry  
- Ridge/valley structure  
- Slope transitions  

But **not** absolute elevation in meters.

This limitation is *inherent* to the dataset and architecture, and fully consistent with the literature.

---

## **6. Differences from Original ImageToDEM Pipeline**

| Component | ImageToDEM (Paper) | Our Downstream Task |
|----------|---------------------|---------------------|
| Input | 3-channel RGB | 4-channel → 3-channel RGB |
| Training | GAN (Generator + Discriminator) | **Inference-only**, no discriminator |
| Loss | L1 + adversarial | None (frozen model) |
| Output | Normalized DEM | Same |
| Purpose | Standalone height estimation | Cloud-removal downstream evaluation |