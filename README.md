**Downstream Part: Evaluating Terrain-Information Recovery via Diffusion-Based Cloud Removal**  

**Short Description:**  
This downstream task evaluates how well our diffusion-based cloud-removal model restores terrain-relevant information in satellite imagery. Using the generator from the ImageToDEM cGAN model (Panagiotou et al., 2020), we generate DEMs from three RGB inputs: the original cloudy image, our cloud-removed prediction, and the ground-truth clean image. By comparing how closely each predicted DEM matches the DEM generated from the clean RGB, we quantify how much terrain structure is recovered after cloud removal.

**What It Does**  
Our downstream module measures **DEM height-field recovery** using three DEMs: (1) DEM from cloudy RGB, (2) DEM from our diffusion-cleaned RGB, and (3) DEM from the ground-truth clean RGB, which serves as the **baseline reference** for terrain structure. All RGB inputs are normalized to the $[-1,1]$ range required by the pretrained ImageToDEM U-Net. We then compute MAE between each DEM and the clean baseline DEM. DEM(cloudy) shows how clouds destroy height information, DEM(clean) represents the ideal terrain signal, and DEM(pred) shows how well our diffusion model restores it. Because the DEM generator is frozen, improvements reflect pure gains from cloud removal rather than changes in the DEM model.

![Pipeline Overview](Images/Downstream_Architecure.png)

**Evaluation:**

We evaluate DEM height-field consistency using MAE between DEM(clean → DEM) and two alternatives:  
**DEM(cloudy → DEM)** — terrain estimation from cloudy RGB  
**DEM(pred → DEM)** — terrain estimation from diffusion-cleaned RGB  

Lower MAE indicates better terrain accuracy.  
Improvement is computed as:

$ \text{Improvement} = 1 - \frac{\text{MAE}_{\text{pred→clean}}}{\text{MAE}_{\text{cloudy→clean}}} $

**Downstream DEM Comparison Table**

| Dataset Split | MAE (Cloudy → Clean) ↓ | MAE (Pred → Clean) ↓ | Improvement ↑ |
|--------------|-------------------------|-----------------------|----------------|
| **Validation** | 0.215648 | 0.071100 | **67.0%** |
| **Test**       | 0.217131 | 0.074932 | **65.5%** |

![Pipeline Overview](Images/Downstream_Outcome.png)
This show that our diffusion-based cloud removal leads to a **65–67% reduction in DEM height estimation error**, demonstrating that the restored RGB imagery retains substantially more terrain-relevant information than raw cloudy imagery. Since the DEM model is frozen, all improvements come strictly from visual restoration.