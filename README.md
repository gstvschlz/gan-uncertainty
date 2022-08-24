<h1 align="center">
    Modeling reservoir uncertainty with Deep Generative Networks
</h1>

  <p align="center">
  <a href="#objective">Objective</a> •
  <a href="#results">Results</a> •
  <a href="#usage">Usage</a> •
  </p>

  <h2 id="objective" > 🎯 Objectives </h2>

  The idea of this paper is to combine the strength of the GAN to learn spatial patterns with the benefits of the SNESIM algorithm to use many types of information for conditioning. The result is a hybrid workflow with two main steps. The first step consists of creating a dataset of training images using GAN. The second step resumes building geostatistical models using the training images generated previously and the conditioning data.

<h2 id="results" > Results and discussion</h2>

  
<h2 id="usage" > 👷 Usage </h2>

Pre-requisites to run the script included in the `requirements.txt` file .

  ```bash
  git clone https://github.com/algocompretto/gan-for-mps.git
  
  # Activates the environment and installs prerequisites
  cd gan-for-mps/ && python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
<br><br>

<h3 id="usage-snesim" > Running SNESIM simulations </h3>
In the project folder, navigate to the `SNESIM` folder, and then execute the script with:

  ```bash
  python3 snesim.py --arguments
  ```

| Argument name    | Description                                                                                    |
|------------------|------------------------------------------------------------------------------------------------|
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

<h3 id="usage-gan" > Running the proposed workflow </h3>
<h4>Training</h4>

<h4>Sampling with pre-trained model</h4>


<h2 id="citation" > ✍️ Citation </h2>
If you use our code for your own research, we would be grateful if you cite our publication
[LinkToBeInserted](https://github.com/)

```
@article{test2023,
	title={Use of Generative Adversarial Networks to incorporate the Training Image Uncertainty in Multiple-Point Statistics Simulation},
	author={Scholze, Gustavo and Bassani, Marcel},
	journal={-},
	year={2023}
}
```