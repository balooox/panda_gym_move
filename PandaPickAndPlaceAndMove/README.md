# Custom Panda Gym Environments

Students project at the University of Applied Sciences Dresden (HTW Dresden). The goal
was to customize an existing reinforcement learning problem and to successfully train a model.
The base for this project was the panda-gym repository (https://github.com/qgallouedec/panda-gym).

Models were successfully trained with the TQC algorithm and partly also with the SAC algorithm. We used 
stable-baselines3 and sb3-contrib to integrate the algorithms. 

We have implemented two custom gym environments:
- PandaPickAndPlaceAndThrow-v1
- PandaPickAndPlaceAndMove-v1

This repository includes trained models for the environments, available as a zip-file. 
More information at the usage section.

## Installation 

This project was executed in a conda environment. Under Windows we used PyCharm as an execution IDE, 
under Linux (Ubuntu) you can execute the code throw the CLI. 

```bash
conda create --name name python=3.10
conda activate name
conda install --channel conda-forge stable-baselines3
conda install --channel conda-forge tensorboard
pip install panda-gym==2.0
pip install sb3-contrib
```

See the requirements.txt for more information

## Usage

### showcase.py 
To get an overview over the custom environments, you can run the script showcase.py !

```bash
python ./showcase.py env
# For example
python ./showcase.py PandaPickAndPlaceAndThrow-v1
```

### train.py
You can train a model with the train.py script. Available algorithms are TQC and 
SAC ( but you can extend the code and include different algorithms available
through stable-baselines3 and sb3_contrib, like PPO etc... ).

```bash
python ./train env algo amount
# For example
python ./train.py PandaPickAndPlaceAndThrow-v1 TQC 1000000
```

### enjoy.py
After a training, a zip file is saved under the ./trained directory.
You can run enjoy.py two see a visualized result of the training

```bash
python ./enjoy env algo file
# For example
python ./enjoy PandaPickAndPlaceAndThrow-v1 TQC ./benchmark/PandaPickAndPlaceAndThrow-v1/TQC/monitor.zip 
```

### evaluate.py

```bash
python  ./evaluate env algo file
# For example
python ./evaluate PandaPickAndPlaceAndThrow-v1 TQC ./benchmark/PandaPickAndPlaceAndThrow-v1/TQC/monitor.zip
```

## Environments

This custom environment was used for `PandaPickAndPlaceAndMove-v1` and `PandaPickAndPlaceAndThrow-v1` 
![Bullet_Physics_PandaPickAndPlaceAndMove_AdobeExpress](https://user-images.githubusercontent.com/92969814/221543161-c6864244-e082-4d00-bb1a-0d3ed8c66278.gif)


