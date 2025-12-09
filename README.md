**Title: Multi-Temporal Diffusion Model for Cloud Removal in Satellite Imagery**
Members: Sihan Yao & Yuxuan Huang


**Description:**
Clouds significantly hinder the visibility of terrestrial targets in satellite observations. This project tackles the problem by implementing a cloud-removal system built on a diffusion-based generative model conditioned on a multi-temporal CNN-based cloud encoder. The model processes a short sequence of three cloudy 4-channel (RGB + NIR) observations and iteratively denoises a latent input to reconstruct a clean, cloud-free image. The system includes a custom 4-channel UNet denoiser, a temporal cloud encoder, and a backward diffusion sampling pipeline. The forward diffusion formulation borrows conceptual ideas from Liu et al. (2025), and the model is trained on the Sen2-MTC multi-temporal Sentinel-2 dataset developed by Huang & Wu (2022).

![Pipeline Overview](Images/Full_Denoising_Network_Overview.png)

![Pipeline Overview](Images/Cloud_encoder.png)

**What It Does:**
Our program removes clouds from multi-temporal satellite imagery by combining temporal feature extraction with a diffusion-based restoration process. Given three temporally adjacent cloudy observations of the same region, the system extracts spatial descriptors and latent temporal embeddings through a Cloud Encoder. A backward diffusion process is then performed, initializing from a noisy version of the first cloudy frame. At each denoising step, a cloud-conditioned UNet predicts the underlying clean signal by approximating the mean of the forward process. Through hundreds of iterative reverse-diffusion steps, the method reconstructs a cloud-free output image without requiring cloud masks typically needed by supervised cloud-removal pipelines. The framework supports inference, visualization, and evaluation through metrics such as MAE, PSNR, SSIM, and LPIPS. Trained on the Sen2-MTC dataset with ~13,700*4 samples (70% training, 30% val/test, patched into 4), the model demonstrates potential for lightweight (50M parameters for our larger model), generalizable cloud removal and downstream remote-censoring tasks.
![Pipeline Overview](Images/Aggression.png)

![Pipeline Overview](Images/Diffusion_Reverse.png)
**Quick Start:**
After SETUP.md,
Quick run (both pre-computed dataset & pretrained model):
```
jupyter lab Notebooks/Pipeline_final_via_pretrained.ipynb
```
To train locally from scratch, run:
```
jupyter lab Notebooks/Pipeline_final_local.ipynb
```
To evaluate pretrained performance, run:
```
jupyter lab Notebooks/Evaluation.ipynb
```
    ** Note that Evaluation.ipynb defaults to pretrained results data by Model.evaluation.evaluate_over_precomputed.
    ** Switch to Model.evaluation.evaluate_over_loader to perform localized evaluation process.
To generate cloud-free predictions val/test loaders, run:
```
jupyter lab Notebooks/Diffusion_Application_Final.ipynb
```

**Video Links:**
https


**Evaluation:**
We evaluate on 30% of our full Sen2-MTC data combining validation and test sets, it consists of 4101 samples in total.
We reference to state-of-the-art cloud removal models for performance evaluation, where Liu et al.'s paper (2025) provides a detailed listing of model performance comparison.

Liu et al's (2025) comparison table on Sen2-MTC dataset shows:
| Method                | PSNR ↑ 	| SSIM ↑ 	| LPIPS ↓ 	|
| --------------------	| --------- | --------- | ---------	|
| McGAN           		| 17.448 	| 0.513  	| 0.447   	|
| Pix2Pix          		| 16.985 	| 0.455 	| 0.535   	|
| AE               		| 15.100 	| 0.441  	| 0.602   	|
| STNet              	| 16.206 	| 0.427  	| 0.503   	|
| DSen2-CR          	| 16.827 	| 0.534  	| 0.446   	|
| STGAN            		| 18.152 	| 0.587  	| 0.513   	|
| CTGAN        			| 18.308 	| 0.609  	| 0.384   	|
| SEN12MS-CR-TS Net 	| 18.585 	| 0.615  	| 0.342   	|
| PMAA           		| 18.369 	| 0.614  	| 0.392   	|
| UnCRtainTS       		| 18.770 	| 0.631  	| 0.333   	|
| Method           		| PSNR ↑ 	| SSIM ↑  	| LPIPS ↓ 	|
| ---------------------	| ---------	| ---------	| --------	| ------- Diffusion-based Methods Start Here
| DDPM-CR 			    | 18.742 	| 0.614	    | 0.329	    |
| DiffCR  				| 19.150 	| 0.671	    | 0.291 	|
| EMRDM			        | 20.067 	| 0.709 	| 0.255 	| ------- Liu et al's (2025)
| Ours				    | 22.695	| 0.888	    | 0.100	    | ------- MAE: 0.017

Our Encoder-Conditioned-Diffusion based cloud removal model outperforms ALL existing models in terms of the Sen2-MTC dataset.
Due to computational resource limitations and time constraints, we are unable to train, fine-tune or test on other datasets.

**Downstream Part: Evaluating Terrain-Information Recovery via Diffusion-Based Cloud Removal**  

**Short Description:**  
This downstream task evaluates how well our diffusion-based cloud-removal model restores terrain-relevant information in satellite imagery. Using the generator from the ImageToDEM cGAN model (Panagiotou et al., 2020), we generate DEMs from three RGB inputs: the original cloudy image, our cloud-removed prediction, and the ground-truth clean image. By comparing how closely each predicted DEM matches the DEM generated from the clean RGB, we quantify how much terrain structure is recovered after cloud removal.

**What It Does**  
Our downstream module measures **DEM height-field recovery** using three DEMs: (1) DEM from cloudy RGB, (2) DEM from our diffusion-cleaned RGB, and (3) DEM from the ground-truth clean RGB, which serves as the **baseline reference** for terrain structure. All RGB inputs are normalized to the $[-1,1]$ range required by the pretrained ImageToDEM U-Net. We then compute MAE between each DEM and the clean baseline DEM. DEM(cloudy) shows how clouds destroy height information, DEM(clean) represents the ideal terrain signal, and DEM(pred) shows how well our diffusion model restores it. Because the DEM generator is frozen, improvements reflect pure gains from cloud removal rather than changes in the DEM model.

![Pipeline Overview](Images/Downstream_Architecture.png)

**Quick Start:**
To test our downstream example, run
```
jupyter lab Downstream/Downstream_Task_Final.ipynb
```
**Evaluation:**

We evaluate DEM height-field consistency using MAE between DEM(clean → DEM) and two alternatives:  
**DEM(cloudy → DEM)** — terrain estimation from cloudy RGB  
**DEM(pred → DEM)** — terrain estimation from diffusion-cleaned RGB  

Lower MAE indicates better terrain accuracy.  
Improvement is computed as:

$Improvement = 1 - MAE_{pred\_clean} / MAE_{cloudy\_clean}$

**Downstream DEM Comparison Table**

| Dataset Split | MAE (Cloudy → Clean) ↓ | MAE (Pred → Clean) ↓ | Improvement ↑ |
|--------------|-------------------------|-----------------------|----------------|
| **Validation** | 0.215648 | 0.071100 | **67.0%** |
| **Test**       | 0.217131 | 0.074932 | **65.5%** |

![Pipeline Overview](Images/Downstream_Outcome.png)
This show that our diffusion-based cloud removal leads to a **65–67% reduction in DEM height estimation error**, demonstrating that the restored RGB imagery retains substantially more terrain-relevant information than raw cloudy imagery. Since the DEM model is frozen, all improvements come strictly from visual restoration.

**Individual Contributions**
Sihan Yao:
Yuxuan Huang