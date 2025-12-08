Title: Multi-Temporal Diffusion Model for Cloud Removal in Satellite Imagery
Members: Sihan Yao & Yuxuan Huang


Description:
Clouds significantly hinder the visibility of terrestrial targets in satellite observations. This project tackles the problem by implementing a cloud-removal system built on a diffusion-based generative model conditioned on a multi-temporal CNN-based cloud encoder. The model processes a short sequence of three cloudy 4-channel (RGB + NIR) observations and iteratively denoises a latent input to reconstruct a clean, cloud-free image. The system includes a custom 4-channel UNet denoiser, a temporal cloud encoder, and a backward diffusion sampling pipeline. The forward diffusion formulation borrows conceptual ideas from Liu et al. (2025), and the model is trained on the Sen2-MTC multi-temporal Sentinel-2 dataset developed by Huang & Wu (2022).


What It Does:
Our program removes clouds from multi-temporal satellite imagery by combining temporal feature extraction with a diffusion-based restoration process. Given three temporally adjacent cloudy observations of the same region, the system extracts spatial descriptors and latent temporal embeddings through a Cloud Encoder. A backward diffusion process is then performed, initializing from a noisy version of the first cloudy frame. At each denoising step, a cloud-conditioned UNet predicts the underlying clean signal by approximating the mean of the forward process. Through hundreds of iterative reverse-diffusion steps, the method reconstructs a cloud-free output image without requiring cloud masks typically needed by supervised cloud-removal pipelines. The framework supports inference, visualization, and evaluation through metrics such as MAE, PSNR, SSIM, and LPIPS. Trained on the Sen2-MTC dataset with ~13,700*4 samples (70% training, 30% val/test, patched into 4), the model demonstrates potential for lightweight (50M parameters for our larger model), generalizable cloud removal and downstream remote-sensing tasks.


Quick Start:
aaa

Video Links:
https


Evaluation:
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

Individual Contributions
Sihan Yao:
Yuxuan Huang