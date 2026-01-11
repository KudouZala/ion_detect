[English](./README.md) | [简体中文](./README.zh-CN.md)  
[![Project Page](https://img.shields.io/badge/🌐%20Project%20Page-1f6feb?logo=github&logoColor=white&labelColor=161b22&style=flat-square)](https://kudouzala.github.io/ion_detect_page/)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face%20Dataset-FFD21E?logo=huggingface&logoColor=black&labelColor=fff3b0&style=flat-square)](https://huggingface.co/datasets/KudouZala/PEM_electrolyzer-ion_detect)




# File Description

- The `data` folder contains the raw voltage and impedance measurement data. `firecloud` and `gamry` are the impedance measurement devices.Each experiment's details.txt file describes the materials and procedures used in the experiment.If an edx folder exists, then it contains the edx data for this experiment.
- The `datasets` folder contains datasets for machine learning training.
- The `logs` folder stores runtime log files.
- The `output` folder contains trained models and inference outputs, including `Attn_heatmap`, `IG`, `Saliency`, and ML analysis results.
- The `scripts` folder stores all source code.


The experimental data and dataset will be released upon acceptance.


# Usage Instructions

This repository is designed to accompany the research paper and allows full reproduction of the workflow.
If you wish to use your own `data` for AutoEIS fitting and training, please organize your files following the same format under the `data` folder. Then, convert the raw data into training-ready format in the `datasets` folder as described below, and start training.
It is recommended to first reproduce the entire process to fully understand how the code works.

---

## Installation and Configuration

If you need to perform EIS fitting, you must install `autoeis` and `julia`.  
The AutoEIS source repository: https://github.com/AUTODIAL/AutoEIS  
Thanks to the author for open-sourcing their code.  
If you cloned this repo directly, it already includes `autoeis`.  
If you do not need EIS fitting and only plan to perform machine learning training, you can skip the installation of `autoeis` and `julia`.  
We recommend running everything on **Ubuntu 22.04**, otherwise certain parts (especially EIS fitting and training scripts) may report errors.

### Environment Setup

1. Create and activate a Conda environment:

   ```bash
   conda create -n ion_detect python=3.10
   conda activate ion_detect
   ```

2. Install PyTorch that matches your CUDA version. (CUDA 12.1 tested and confirmed working.)

   ```bash
   conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

3. Clone this project and install dependencies:

   ```bash
   git clone https://github.com/KudouZala/ion_detect.git
   cd ion_detect
   pip install -r requirements.txt
   ```

---

### Install Julia and AutoEIS

1. Download Julia from https://julialang.org/downloads/ (use the LTS version) and extract it to `/opt` (or another directory):

   ```bash
   tar -xvzf julia-1.10.10-linux-x86_64.tar.gz
   sudo mv julia-1.10.10 /opt/
   ```

2. Create a symbolic link to make `julia` command available:

   ```bash
   sudo ln -s /opt/julia-1.10.10/bin/julia /usr/local/bin/julia
   echo 'export PATH="/opt/julia-1.10.10/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   ```

3. Verify Julia installation:

   ```bash
   julia --version
   ```

4. Install `autoeis`:

   ```bash
   cd autoeis/
   conda activate ion_detect
   pip install -e .
   ```

---

## Using Raw Data

The `data` folder contains the raw voltage and impedance measurements.  
Download link: https://huggingface.co/datasets/KudouZala/PEM_electrolyzer-ion_detect
After downloading, extract to the `data` folder following the structure `/ion_detect/data/校内测试/xxx测试`.  
The data includes both `firecloud` and `gamry` measurement devices.  
The `scripts` folder under `data` contains tools for impedance fitting, voltage analysis, and impedance plotting.

### View Relative Voltage Variation Over Time


Use `ion_detect/data/scripts/eis_code/v_t_plot_code/v_t_relative_compare_xx.py` as a template, fill in `group_folders_firecloud` and `group_folders_gamry`, then run:

```bash
python data/scripts/eis_code/v_t_plot_code/v_t_relative_compare_plot.py 
```
![Voltage variation over time](./github_png/voltage_0_t.png)

Compare voltages before and after H2SO4 recovery:

```bash
python data/scripts/eis_code/v_t_plot_code/v_t_relative_compare_all_renew.py
```
![Voltage variation after H2SO4 recovery](./github_png/voltage_0_t_renew.png)
The output images are saved in `ion_detect/data/volt_t_plot`.

---

### View Nyquist Impedance Variation Over Time


Use `ion_detect/data/scripts/eis_code/v_t_plot_code/nyquist_plot_zhiyun_and_gamry_xx.py` as a template, fill in the required folder paths, then run:

```bash
python data/scripts/eis_code/nyquist_plot_code/nyquist_plot_zhiyun_and_gamry_plot.py
```
![Nyquist impedance variation over time](./github_png/impe_0_t_ca.png)


Similarly, for Bode plots:
```bash
python data/scripts/eis_code/nyquist_plot_code/bode_plot_zhiyun_and_gamry_plot.py
```
![Bode impedance variation 1](./github_png/bode_0_t_ca1.png)  
![Bode impedance variation 2](./github_png/bode_0_t_ca2.png)

---

### Impedance Fitting

```bash
python data/scripts/all_impedance_fit.py > ../logs/$(date +%Y%m%d).log 2>&1
```
You can customize the equivalent circuit and the data to fit in `all_impedance_fit.py`.  
Logs are saved to `ion_detect/data/logs`, and fitting results to `/ion_detect/data/eis_fit_results/<date>/`.


### Using Your Own Test Data for Impedance Fitting

If you wish to use your own test data, add it under the `校内测试` folder following the same structure.  
Copy the `code` folder (and its scripts) from an existing folder such as `data/校内测试/20250103_无离子污染测试/code` into your new folder.  
 `python run_data_exchange_code_gamry_single.py or run_data_exchange_code_firecloud_single.py` for your data format, then run it to generate `output_txt`, `output_csv`, and `output_xlsx` folders with transferred data.  
Finally, add your folder path into the main program of `all_impedance_fit.py` for fitting.  
The fitting results will be saved in `ion_detect/data/eis_fit_results/<date>/`.

---

### Export Impedance Analysis Results

To sort fitted results in the order `RO, R1, R2, R3` (Ohmic → Low-frequency → Mid-frequency → High-frequency):

```bash
python data/scripts/excel_code/excel_PnPw_sequence_all.py  # Modify folder_path = "20250724" to match your eis_fit_results folder name
```
Each subfolder will contain `_sorted.xlsx` and `_sorted.png` files.  
![Sorted fitting result](./github_png/fit_sorted.png)

---

### Impedance Fitting Result Analysis

Compare variations of equivalent circuit parameters for each ion type:

```bash

python data/scripts/fit_res_analysis.py  # Modify date_folder = "20250723" to match your eis_fit_results folder name
```
Results will appear under `/ion_detect/data/eis_fit_analysis_results`, e.g. `_20250723.xlsx` and `_20250723.png`, showing parameter changes from 0–6h for each ion type.

![2RC fitting](./github_png/impe_fit_change_2rc.png)  
![3RC fitting](./github_png/impe_fit_change_3rc.png)

---

### Impedance Fitting Result Analysis – H2SO4 Recovery Comparison

```bash
python data/scripts/fit_res_analysis2.py  # Modify date_folder = "20250723" to match your eis_fit_results folder name
```
Results appear in `/ion_detect/data/eis_fit_analysis_results`, e.g. `_20250723.xlsx` and `_20250723.png`, showing parameter variations before, during, and after H2SO4 recovery.

![2RC fitting H2SO4](./github_png/impe_fit_change_2rc_H2SO4.png)  
![3RC fitting H2SO4](./github_png/impe_fit_change_3rc_H2SO4.png)

---

### Machine Learning Training
If you want to use your own data to convert into training data, please read steps one and two; if you simply want to reproduce the paper, please skip to step three.
1. Organize the `gamry` or `firecloud` data in the same folder format in ion_detect/data/校内测试/..., then add your folder paths into `label_machine_learning_excel_export_gamry_range.py` or `label_machine_learning_excel_export_firecloud_range.py`.

2. Run the following commands to generate formatted Excel files:

   ```bash
   python data/scripts/excel_code/label_machine_learning_excel_export_gamry_range.py
   python data/scripts/excel_code/label_machine_learning_excel_export_firecloud_range.py
   ```

   The generated Excel files will appear under `/ion_detect/data/校内测试/数据整理_range`.  
   Select your desired training data and place it into the `ion_detect/datasets/...` folder for training.  

   Naming conventions:  
   - `_ion_` → data with ion contamination  
   - `_ion_column_` → data before ion contamination  
   - `_ion_column_renew_H2SO4_` → data after H2SO4 recovery  

   You can create your own training scripts under `/ion_detect/scripts/machine_learning_code/`, e.g. `20251213b.yaml`, and run training with:
   ```bash
   python scripts/machine_learning_code/main.py --config 20251213b.yaml --train
   ```
   Testing:
   ```bash
   python scripts/machine_learning_code/main.py --config 20251213b.yaml --test
   ```

3. Example training/testing commands:

   ```bash
   # Recommended to set 20251213b.yaml `test_folder: "datasets/datasets_for_all_test"`, if this folder name is datasets_for_all_test => auto split train- and test-datasets(include all ion types).  
   python scripts/machine_learning_code/main.py --config 20251213b.yaml --train > ./logs/20251213b.log 2>&1
   # see loss curve
   tensorboard --logdir output/trained_model_save/ --port 6006 --bind_all
   # use the Browser:http://127.0.0.1:6006/
   ```

   Model checkpoint search and evaluation:
   ```bash
   python scripts/machine_learning_code/main.py --config 20251213b.yaml --search > ./logs/20251213b_search.log 2>&1
   ```

   Results are saved under `/ion_detect/output/trained_model_save/20251213b/`.  
   ![Search checkpoints accuracy](./github_png/search_checkpoints_accuracy_by_epoch.png)

   Then you can name the best model "trained_model_epoch_final.pth",and to use it to do the after things:
   ```
   # loads "trained_model_epoch_final.pth" model by default
   python scripts/machine_learning_code/main.py --config 20251213b.yaml --test > ./logs/20251213b_test.log 2>&1  
   ```
   Results are saved under `/ion_detect/output/inference_results/20251213b/`.  
   Multi-process debug logs are stored in `/ion_detect/scripts/machine_learning_code/debug_logs`.


---

### AI-Assisted Ion Effect Analysis
1. To visualize attn_heatmap saliency IG across different 2ppm ions in 0-6h:
```bash
#set the config yaml : num_time_points: 4,and then train the model:
python scripts/machine_learning_code/main.py --config 20251213b.yaml --train
python scripts/machine_learning_code/main.py --config 20251213b.yaml --search
#search the best model and name it :"trained_model_epoch_final.pth"(the models are stored in ion_detect/output/trained_model_save/20251213b/...)
#remove the files in "/ion_detect/output/inference_results/20251213b" ,set 20251213b.yaml: `test_folder: "datasets/datasets_for_0_6_2ppm"
python scripts/machine_learning_code/main.py --config 20251213b.yaml --test
python scripts/ml_analysis_code/csv_plot.py --load_run=20251213b
```
This visualizes test results. (Make sure to run `python main.py --config 20251213b.yaml --test` first.)  
The model generates intermediate prediction values and visualization results including `attn_heatmap`, `saliency`, and `IG` under `ion_detect/output/inference_results/20251213b/...`.
![AI analysis PNG- attn/IG/Saliency](<github_png/20241006_2ppm铬离子污染测试_新版电解槽_ion_gamry_[0, 2, 4, 6]_pred4_attribution_plot.png>)


2. To visualize variations of the 5 influence factors across different 2ppm ions in 0-6h:

```bash
#remove the files in "/ion_detect/output/inference_results/20251213b" ,set 20251213b.yaml `test_folder: "datasets/datasets_for_0_6_2ppm"
python scripts/machine_learning_code/main.py --config 20251213b.yaml --test
python scripts/ml_analysis_code/csv_plot_param.py --load_run=20251213b
```
This generates visualization for ion-specific influence factors under `ion_detect/output/inference_results/20251213b`.
![5 influence factors across different 2ppm ions in 0-6h](github_png/20251213b_ions_param_plot.png)

3. To visualize 0–6h initial-state changes of 5 parameters (requires datasets covering `[0,2,4,6]` and `[6,8,10,12]`):

```bash
#remove the files in "/ion_detect/output/inference_results/20251213b" ,set 20251213b.yaml `test_folder: "datasets/datasets_for_range_ion_0_12_2ppm"
python scripts/machine_learning_code/main.py --config 20251213b.yaml --test 
python scripts/ml_analysis_code/csv_plot_param2.py --load_run=20251213b
```
These plots can be compared with impedance fitting variations.

![AI ion analysis](./github_png/ai_ion.png)

---

### AI-Assisted H2SO4 Recovery Analysis

```bash
#remove the files in "/ion_detect/output/inference_results/20251213b" ,set 20251213b.yaml `test_folder: "datasets/datasets_for_all_2ppm"

python main.py --config 20251213b.yaml --test 
python scripts/ml_analysis_code/h2so4_analysis.py --load_run=20251213b
```
This visualizes 5 key factor changes across pre-contamination (0h), post-contamination (6h), and recovered states.  
Requires datasets containing `[0,2,4,6]` and `[6,8,10,12]`.  
The plots can be compared with impedance fitting results for the same three time points.

![AI H2SO4 recovery analysis](./github_png/ai_H2SO4.png)


# Acknowledgement
This project includes code adapted from AutoEIS
(https://github.com/AUTODIAL/AutoEIS),
licensed under the MIT License.

Copyright (c) AutoEIS contributors.