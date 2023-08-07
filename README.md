<h1 align="center">
   Use of Generative Adversarial Networks to incorporate the Training Image Uncertainty in Multiple-Point Statistics Simulation
</h1>

  <p align="center">
  <a href="#objective">Objective</a> •
  <a href="#results">Results</a> •
  <a href="#usage">Usage</a> •
  </p>

  <h2 id="objective" > 🎯 Objectives </h2>

Multiple-Point Geostatistical (MPS) methods have been successfully applied to build numerical models with curvilinear features using several sources of information. Even though traditional algorithms reproduce the spatial pattern of the variogram models, they fail describing curvilinear features - which came from a conceptual model of the underlying geology provided by the expert geologist.  *However, there is hope - MPS new methods reproduces these patterns we wish to replicate.*

In this work, we chose the SNESIM algorithm (Strebelle, 2002) for three reasons: (a) because it is a method widely used by the community; (b) its parameters are intuitive and interpretable; (c) SNESIM algorithm is freely available (Remy and Boucher, 2009).

The training image is uncertain as the actual spatial pattern is unknown. 
This uncertainty is even more pronounced at the exploration stage when little information is available (Pyrcz and Deutsch, 2014). Considering the uncertainty of the input parameters improves the assessment of the space of uncertainty, Pyrcz and Deutsch (2014) recommend using a scenario-based approach to incorporate the lack of confidence of the parameters in the simulations.

The idea is to merge the generative model adeptness to learn spatial patterns with the benefits of the SNESIM algorithm to use many types of information for conditioning. The outcome is a hybrid workflow with two main steps: (a) creating a dataset of training images using the generative model; (b) building geostatistical models using the synthetic TI and the existing conditional data.

<h2 id="results" > Results and discussion</h2>

  
<h3 id="usage" > 👷 Usage </h2>

Pre-requisites to run the script included in the `requirements.txt` file .

  ```shell
  git clone https://github.com/algocompretto/gan-uncertainty.git
  
  # Activates the environment and installs prerequisites
  cd gan-uncertainty/ && python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
<br><br>

<h3 id="usage-snesim" > Running SNESIM simulations </h3>
In the project folder, navigate to the `SNESIM` folder, and then execute the script with:

  ```shell
  python3 snesim.py --arguments
  ```

| Argument name    | Description                                                                                    |
| ---------------- | ---------------------------------------------------------------------------------------------- |
| `--samples_path` | The samples path. The file should contain data in the following format: `x`,`y`,`z`,`facies` . |
| `--ti_path `     | The training image path in GSLIB format.                                                       |
| `--par_path`     | Path to the parameter file with all information related to the simulation process itself.      |
| `--exe_path`     | The `snesim.exe` file path.                                                                    |
| `--output_path`  | Path to the output file.                                                                       |
| `--realizations` | Number of realizations to be done.                                                             |
| `--max_cond`     | The maximum amount of points to use in the conditioning process.                               |
| `--min_cond`     | The minimum amount of points to use in the conditioning process.                               |
| `--seed`         | The initial seed for the simulation.                                                           |
| `--plot`         | A boolean value for whether you want to plot/save the results or not.                          |

<h2 id="usage-gan" > Running the proposed workflow </h3>
<h3>Training</h4>
If you wish to train a new model on unseen data, you can follow the next steps:

```shell
cd generative_model/
python3 gan.py
```
The training will get all the information on hyperparameters from the `parameters.yaml` file

| Argument name    | Description                                                                    |
| ---------------- | ------------------------------------------------------------------------------ |
| `output_dir`     | The output directory for the augmented images.                                 |
| `training_image` | The training image path in `.png` format.                                      |
| `checkpoint`     | The checkpoint folder which the models will be stored.                         |
| `sample_images`  | The folder where the sampled examples from the network will be saved.          |
| `num_channels`   | Number of channels in the image                                                |
| `latent_dim`     | The latent dimension vector size representing the features.                    |
| `learning_rate`  | The learning rate for the Adam optimizers                                      |
| `images_path`    | The output directory for the windowed images.                                  |
| `batch_size`     | The batch size for the training step.                                          |
| `num_workers`    | The number of workers to load the dataset.                                     |
| `num_epochs`     | The number of epochs for training step.                                        |
| `cuda`           | A boolean value for whether you want to use the CUDA device or not.            |
| `n_critic`       | The number of steps to train the Critic after `n` iterations of the Generator. |
