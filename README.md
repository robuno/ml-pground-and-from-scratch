# ml-playground-and-projects-from-scratch

Here I collect some basic ML, NN models that I wrote from scratch to learn how they works. Some of them are repetitions of codes already written in the tutorials, with new comments and code simplifications made on them. For credits, see below:


### Simple Diffusion Model Analysis (Unidimensional Data with 4 categories)
Change of data according to noise addition and reverse process
![Change of data according to noise addition and reverse process](https://raw.githubusercontent.com/robuno/ml-pground-and-from-scratch/main/figures/simple_diff/data_gen_per_time.png)

Data Change by adding Gaussian Noise
![Data Change by adding Gaussian Noise](https://raw.githubusercontent.com/robuno/ml-pground-and-from-scratch/main/figures/simple_diff/histogram_gaussian_noises.png)

Data Change by adding Gaussian Noise
![Mean Change wrt. Time Stamps](https://raw.githubusercontent.com/robuno/ml-pground-and-from-scratch/main/figures/simple_diff/mean_per_time.png)



### CLIP
-  Shariatnia, M. (2021, April 7). Simple Implementation of OpenAI CLIP model: A Tutorial. Medium. https://towardsdatascience.com/simple-implementation-of-openai-clip-model-a-tutorial-ace6ff01d9f2

![Correlation Matrices of CLIP](https://raw.githubusercontent.com/robuno/ml-pground-and-from-scratch/main/figures/clip_corr_matrices2.png)

### Autoencoder & Variational Autoencoder (VAE)
- Implemented from scratch with 2 model version of AE's. First one is very simple AE with 2 conv layers, second AE is more complex taken from [Sebastian Raschka](https://github.com/rasbt/stat453-deep-learning-ss21/tree/main/L16)'s tutorial.

![t-SNE Plot for AE's Encoded Images](https://raw.githubusercontent.com/robuno/ml-pground-and-from-scratch/main/figures/ae-mnist-tsne-60000.png)

### NanoGPT

-  [NanoGPT lecture by Andrej Karpathy](https://github.com/karpathy/ng-video-lecture)

Example output with Nazim Hikmet's poems:
```
Lambayi girinde sordu ekrep olanmaya geldik,
yukarda bütün senden çelişirsiz,
ölmek istemiyordu yine dasina
Niyida bir ada bir hanga verdi
Madrid kalbimi ben
Okranliğini bilmez orani

Gözler ağizliğindan tütün adin
Bile birakiz vurur bir rahat
Budurunla da balik dururdu
Yüz başlari suyu kendil yüzünü
orta yildizlar anasini

Topal Meryem'in üstüne  duyulcular bir eferrun,
Yildizlarinci asirlarda vurursun.
Birden mavi gözleri haber gibi bir adam gibi yaprakla azi bulunun atmadi.
```
<!-- <img width="400" height="600" src="https://raw.githubusercontent.com/robuno/ml-pground-and-from-scratch/main/figures/n_gpt_losses1_nazim.png"> -->
![Train/Val Losses of Models](https://raw.githubusercontent.com/robuno/ml-pground-and-from-scratch/main/figures/n_gpt_losses1_nazim.png)



### Language Detection
<!-- <img width="400" height="600" src="https://raw.githubusercontent.com/robuno/ml-pground-and-from-scratch/main/figures/n_gpt_losses1_nazim.png"> -->
![Conf. Matrix Language Detection N=1](https://raw.githubusercontent.com/robuno/ml-pground-and-from-scratch/main/figures/detect_lang_n1.png)
![Conf. Matrix Language Detection N=2](https://raw.githubusercontent.com/robuno/ml-pground-and-from-scratch/main/figures/detect_lang_n2.png)
![Conf. Matrix Language Detection N=3](https://raw.githubusercontent.com/robuno/ml-pground-and-from-scratch/main/figures/detect_lang_n3.png)

