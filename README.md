This is the COMPSCI-372 Final Project of Sihan Yao & Yuxuan Huang.
# Downstream Task: Evaluating Cloud Removal via DEM Height Estimation

## 1. Goal

The downstream DEM experiment is designed to answer one question:

> Does removing clouds with our multi-temporal diffusion model make **height estimation** from satellite images more accurate?

We use a **pretrained ImageToDEM U-Net** (Panagiotou et al., 2020) as a fixed evaluator.  
It maps RGB tiles to DEM tiles, and we do **not** retrain it. Our diffusion model only changes the **input images**:

- **Baseline:** DEM predicted from *cloudy* RGB
- **Ours:** DEM predicted from *cloud-removed* RGB

If our cloud removal is useful, the DEM errors should be lower in the second case.

---

## 2. Pipeline

Each sample is a 4-channel patch of size 128×128×4:

- Channels 0–2: RGB
- Channel 3: cloudiness or NIR (ignored by the DEM model)

**Steps:**

1. **Drop the 4th channel**  
   We keep only RGB: shape becomes 3×128×128.

2. **Normalize RGB to [-1, 1]**  
   This matches the normalization used by ImageToDEM’s pix2pix generator.

3. **Pass through pretrained generator**  
   The generator outputs a 1-channel DEM tile (128×128×1).

4. **Compare to ground-truth DEM with MAE**  
   We compute the error separately for:
   - cloudy input (baseline)
   - cloud-removed input (ours)

---

## 3. DEM Model and Height Interpretation

In the ImageToDEM paper, DEM generation is modeled as a mapping  

- $G : X \to Y$,  
  where $X$ is the space of RGB images and $Y$ is the space of DEMs. :contentReference[oaicite:0]{index=0}  

The network output is **normalized**:

- Inputs and DEMs are scaled so that predicted heights lie in a fixed range (e.g. $[-1, 1]$), because training on unbounded heights would require unrealistic amounts of data. :contentReference[oaicite:1]{index=1}  

To recover true heights in meters one would need the **true minimum and maximum altitude** $(H_{\min}, H_{\max})$ for each validation region and then apply a linear rescaling such as

- $h_{\text{true}} \approx \frac{h_{\text{pred}} + 1}{2} \,(H_{\max} - H_{\min}) + H_{\min}$

As the paper notes, without knowing $H_{\min}$ and $H_{\max}$ for each tile, the model can only provide **relative elevation structure**, not absolute height in meters. :contentReference[oaicite:2]{index=2}  

This is exactly the regime of our experiment:  
we evaluate how well the cleaned images preserve *relative terrain geometry* as measured by a pretrained DEM network.

---

## 4. Error Metrics and Improvement Formula

For a DEM tile of size $H \times W$ with prediction $\hat{D}$ and ground truth $D$:

- Mean Absolute Error (MAE):  
  $MAE = \frac{1}{HW} \sum_{i,j} \left| \hat{D}_{ij} - D_{ij} \right|$

To quantify the benefit of cloud removal we use:

- Relative improvement (fractional error reduction):  
  $Improvement = 1 - \dfrac{MAE_{clean}}{MAE_{cloudy}}$

where

- $MAE_{cloudy}$ is the error when the DEM model sees cloudy RGB  
- $MAE_{clean}$ is the error when it sees cloud-removed RGB

---

## 5. Results: Cloud Removal Improves DEM Prediction

### Validation Set

- Cloudy → GT DEM (baseline):  
  $MAE_{cloudy} = 0.215648$
- Cloud-removed (ours) → GT DEM:  
  $MAE_{clean} = 0.071100$

Relative improvement on validation:

- $Improvement_{val} = 1 - 0.071100 / 0.215648 \approx 0.6703$  
  → about **67.0% error reduction**

---

### Test Set

- Cloudy → GT DEM (baseline):  
  $MAE_{cloudy} = 0.217131$
- Cloud-removed (ours) → GT DEM:  
  $MAE_{clean} = 0.074932$

Relative improvement on test:

- $Improvement_{test} = 1 - 0.074932 / 0.217131 \approx 0.6549$  
  → about **65.5% error reduction**

---

## 6. Interpretation

1. **Clouds harm DEM estimation.**  
   When the DEM model sees cloudy RGB, MAE is around 0.216–0.217.

2. **Our cloud-removed images are much more informative.**  
   When we feed the *same* DEM model our diffusion-cleaned RGB, the MAE drops to about 0.071–0.075.

3. **Quantitatively, this is a large effect.**  
   Around **67% error reduction on validation** and **65.5% on test**, using a DEM network that was never trained on our cleaned images.

4. **This connects cloud removal to height estimation theory.**  
   The DEM network only sees local spectral cues and outputs normalized relative heights. The fact that its relative height predictions improve so much after cloud removal shows that our diffusion model is restoring terrain-relevant information, not just making images look nicer.

**Conclusion:**  
Our multi-temporal diffusion model for cloud removal substantially improves downstream DEM height estimation, providing strong evidence that cloud-free reconstructions produced by our method are more useful for real remote-sensing workflows than the original cloudy imagery.
