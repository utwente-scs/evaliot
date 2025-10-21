# Reproduce the claims from the paper

## Dataset Setup and Evaluation Method

### 1. Download datasets  
Download the following files from [here](https://drive.google.com/drive/folders/1s6kpXD9fSfkK6dnhyxvWnkOOyjOqqbdZ?usp=drive_link):  
- `Experiment-1.zip`  
- `Experiment-2.zip`  


### 2. Extract datasets

Unzip all downloaded files and move the extracted folders into the `artifact/dataset` directory.  

```bash
unzip Experiment-1.zip -d dataset/
unzip Experiment-2.zip -d dataset/
```


### 3. Split Dataset

The dataset contains mixed network traffic from both IoT and non-IoT devices, captured at the router. Since some evaluation methods require traffic in a *per-device* format, we created a script, `split-traffic.py`, which separates the dataset into individual device traffic files based on their MAC addresses.

```bash
cd artifact
python scripts/split-traffic.py <dataset-path> <dataset-path>/devices.txt
```

For convenience and to make it easier to reproduce the experiments, the per-device traffic for these experiments is already provided in the dataset:

- `Experiment-1_per_device.zip`  
- `Experiment-2_per_device.zip`  


## Clone the Re-implemented repositories

### Automatically using _install.sh_

- Run the install script to automatically create the `reimplementations` directory and automatically clone all the implmentations in it.

```bash
./install.sh
```

### Manually clone each repository

- Create a directory called `reimplementations`.
```bash
mkdir reimplementations
```

- Clone the repositories of the evaluation methods to the local directory `reimplementations`.
- The list of available repositories can be found in [EVAL_README](./artifact/framework/eval_modules/EVAL_README.md)

```bash
git clone <repo-link> <local-path>
```

## Run Batch Experiments

- Run the script `run_claim.sh` using the claim number (1, 2, 3, 4) as an argument. The script will automatically run all options for all methods. This process may take some time, so please be patient.

```bash
./run_claim.sh <claim-number>
```

## Run Individual Experiments

### Experiment 1 : IoT vs non-IoT devices

The `Experiment-1` dataset contains 3 directories in it, each corresponding to the 3 sets of data collected of this experiment:
- `Exp-1_iot` : This was collected with only IoT devices connected to the network.
- `Exp-1_iot_mobile` : This was collected with IoT and mobile devices (phone and Ipad) connected to the network.
- `Exp-1_iot_mobile_it`: This was collected with IoT, mobile and IT devices (laptops) connected to the network.


#### Running Experiments with Different Train/Test Combinations (as shown in **TABLE II** in the paper):

- Run experiments with different **train/test** dataset pairs. The six combinations are:

| Experiment | Train Dataset         | Test Dataset            |
|------------|---------------------|------------------------|
| 1          | Exp-1_iot           | Exp-1_iot              |
| 2          | Exp-1_iot           | Exp-1_iot_mobile       |
| 3          | Exp-1_iot           | Exp-1_iot_mobile_it    |
| 4          | Exp-1_iot_mobile    | Exp-1_iot_mobile       |
| 5          | Exp-1_iot_mobile    | Exp-1_iot_mobile_it    |
| 6          | Exp-1_iot_mobile_it | Exp-1_iot_mobile_it    |

---

- The configs for all 6 options are given in the `claims/claim1/` directory. The configs are placed within the method specific directory.
- Run the experiments with the path to the configs.

```bash
cd artifact
python main.py ../claims/claim1/<method-name>/config<num>.yml
```

- Compare the results with the ones given in **TABLE II** of the paper.

---

### Experiment 2 : Static vs Dynamic Environments

The `Experiment-2` dataset contains 2 directories in it, each corresponding to the 2 sets of data collected of this experiment:
- `Exp-2_static` : This was collected in a static setup, where the number of IoT and controlling mobile devices remains unchanged throughout data collection period.
- `Exp-2_dynamic` : This was collected in a dynamic setup, where the number of IoT devices, mobile devices, or both changed during data collection period.

#### Running Experiments with Different Train/Test Combinations (as shown in **TABLE III** in the paper):

Run experiments with different **train/test** dataset pairs. Here are the three combinations:

| Experiment | Train Dataset         | Test Dataset           |
|------------|-----------------------|------------------------|
| 1          | Exp-2_static          | Exp-2_static           |
| 2          | Exp-2_static          | Exp-2_dynamic          |
| 3          | Exp-2_dynamic         | Exp-2_dynamic          |

> **_NOTE:_**  Keep the `use-known-devices` as `yes`.

- The configs for all 3 options are given in the `claims/claim2/` directory. The configs are placed within the method specific directory.
- Run the experiments with the path to the configs.

```bash
cd artifact
python main.py ../claims/claim2/<method-name>/config<num>.yml
```
- Compare the results with the ones given in **TABLE III** of the paper.


---

### Experiment 3 : Scalability Test

We evaluate the method with 2 differents datasets for this experiment:
- [MONIOTR US](https://moniotrlab.khoury.northeastern.edu/publications/imc19/) : 46 devices (*Due to the [data sharing agreement](https://moniotrlab.khoury.northeastern.edu/wp-content/uploads/2019/11/iot-dataset-terms-imc19.pdf), we cannot provide the datasets used to reproduce the paper’s results.*)
- `Exp-2_dynamic` : 23 devices (*We use the `per_device` format of the dataset*)

<br>

For this experiment, we created a script that evaluates the method consecutively with varying numbers of devices, starting from 7.

1. Open the config file of the method to evaluate.
2. Set both training and testing to the same dataset.
3. Run `scripts/run-scalability-test.py` with the config file path as an argument.
4. The config files for each method is given in the `claims/claim3` directory.

```bash
cd artifacts
python scripts/run-scalability-test.py ../claims/claim3/<method-name>/config.yml
```

> **_NOTE:_**  To run this experiment with a different dataset, ensure it is in `per_device` format and includes a `devices.txt` file containing device names and MAC addresses.

- Compare the results with the ones given in **FIGURE II** of the paper.

---

### Experiment 4: Detecting Unknown Devices

We use the `Experiment-2` dataset for this evaluation. 
1. Train with `Exp-2_static` and test with `Exp-2_dynamic`.
2. Set `use-known-devices` to `no`.
3. Set the `threshold` value.
    - We tested with all values between [0.1 - 0.7] and selected the values that minimized overall misclassifications.
4. The config files for each method is given in the `claims/claim4` directory.

> **_NOTE:_**  For the baseline values, run the experiment with `use-known-devices` set to `yes`.

```bash
cd artifact
python main.py ../claims/claim4/<method-name>/config.yml
```

- Compare the results with the ones given in **TABLE IV** of the paper.

---