LearningRacer-rl
======

Overview

This software is able to Self learning your AI RC Car 
by Deep reinforcement learning in few minutes.

![demo](content/demo.gif)

## Description

DIY self driving car like JetBot or JetRacer, DonkeyCar are  learning by supervised-learning.
The method need much labeled data that written human. Running behavior characteristic is determined that data.

Deep reinforcement learning (DRL) is can earned running behavior automatically through interaction with environment.
Do not need sample data that is human labelling.

This is using Soft Actor Critic as DRL algorithm. The algorithm is State of The Art of DRL in real environment.
In addition, using Variational Auto Encoder(VAE) as State representation learning. 
VAE can compress environment information, can speed up learning.


* This method devised by Arrafin
    * [Arrafine's Medium blog post](https://towardsdatascience.com/learning-to-drive-smoothly-in-minutes-450a7cdb35f4)
    * [Arrafine's implementsation for Simulator](https://github.com/araffin/learning-to-drive-in-5-minutes)


* About Soft actor critic
    * [Google AI blog Soft Actor-Critic: Deep Reinforcement Learning for Robotics](https://ai.googleblog.com/2019/01/soft-actor-critic-deep-reinforcement.html)

## Demo

This video is 
JetBot is learning running behavior on road in under 30 minutes. Software is running on Jetson Nano.  

[![](https://img.youtube.com/vi/j8rSWvcO-s4/0.jpg)](https://www.youtube.com/watch?v=j8rSWvcO-s4)


## Setup

### Requirements

* JetBot or JetRacer base image(Recommend latest images)
* tensorflow-gpu=1.14.0
* torch
* torchvision
* OpenCV

### Install

Dependency library install.

#### posix_ipc

```shell
sudo pip3 install posix_ipc
```

#### OpenAIGym 0.10.9

```
sudo apt install -y liblapack-dev scipy
sudo pip3 install gym==0.10.9
```

#### stable-baselines v2.9.0

```
$ sudo pip3 install -U Cython
$ cd ~/ && git clone https://github.com/hill-a/stable-baselines.git -b v2.9.0
# Before install, check below comment.
$ cd stable-baselines/ && sudo python3 setup.py install

Install takes time.
```

1. Must be change setup.py. delete opencv-python from dependencies.


#### clone this repository

```
$ cd ~/ && git clone https://github.com/masato-ka/airc-rl-agent.git
$ cd airc-rl-agent
```

## Usage

### Create VAE Model

1. Collect Environment data as 1k to 10 k images using ```data_collection.ipynb``` or ```data_collection_without_gamepad.ipynb```in ```notebook/utility/jetbot```.
If you use on JetRacer, use```notebook/utility/jetracer/data_collection.ipynb``` . 
2. Learning VAE using ```VAE CNN.ipynb``` on Google Colaboratory.
3. Download vae.torch from host machine and deploy to root directory.

### Check and Evaluation 


Run ```notebooks/util/jetbot_vae_viewer.ipynb``` and Check reconstruction image.
Check that the image is reconstructed at several places on the course.

If you use on JetRacer, Using ```jetracer_vae_viewer.ipynb``` .

* Left is an actual image. Right is reconstruction image.
* Color bar is represented latent variable of VAE(z=32 dim).

![vae](content/vae/vae.gif)


### Start learning

1. Run user_interface.ipynb (needs gamepad).
If you not have gamepad, use ```user_interface_without_gamepad.ipynb```
2. Run train.py

```shell
$ python3 racer.py train -robot jetbot
# If you use on JetRacer, "-robot jetracer". default is jetbot.
```

After few minutes, the AI car starts running. Please push STOP button immediately before the course out. 
Then, after `` `RESET``` is displayed at the prompt, press the START button. Repeat this.

![learning](content/learning.gif)


* train.py options

|Name           | description            |Default                |
|:--------------|:-----------------------|:----------------------|
|-vae(--vae-path)| Specify the file path of the trained VAE model.    | vae.torch             |
|-device(--device)|Specifies whether Pytorch uses CUDA. Set 'cuda' to use. Set 'cpu' when using CPU.| cuda                 |
|-robot(--robot-driver)| Specify the type of car to use. JetBot and JetRacer can be specified.| JetBot              |
|-steps(--time-steps)| Specify the maximum learning step for reinforcement learning. Modify the values ​​according to the size and complexity of the course.| 5000 |
|-s(--save)    | Specify the path and file name to save the model file of the training result.  | model                 |

## Running DEMO

You can running your car without learning. Run below command, The script load vae model and RL model 
and start controll your car.

```shell
$ python3 racer.py demo -robot jetbot
``` 

* demo.py options

|Name           | description            |Default                |
|:--------------|:-----------------------|:----------------------|
|-vae(--vae-path)| Specify the file path of the trained VAE model.    | vae.torch             |
|-model(--model-path|Specify the file to load the trained reinforcement learning model.|model|
|-device(--device)|Specifies whether Pytorch uses CUDA. Set 'cuda' to use. Set 'cpu' when using CPU.| cuda                 |
|-robot(--robot-driver)| Specify the type of car to use. JetBot and JetRacer can be specified.| JetBot              |
|-steps(--time-steps)| Specify the maximum step for demo. Modify the values ​​according to the size and complexity of the course.| 5000 |


## Release note

* 2020/03/08 Alpha release
    * First release.
    
* 2020/03/16 Alpha-0.0.1 release
    * Fix import error at jetbot_data_collection.ipynb.

* 2020/03/23 Beta release
    * VAE Viewer can see latent space.
    * Avoid stable_baseline source code change at install.
    * train.py and demo.py merged to racer.py.
    * Available without a game controller.
    * Fix for can not copy dataset from google drive in CNN_VAE.ipynb

* 2020/03/23 Beta-0.0.1 release
    * Fix VAE_CNN.ipynb (bug #18).

### Running trained model

After training, run the demo.py

In below command, run the demo 1000 steps with model file name is model.

```shell
$ python3 demo.py -robot jetbot -steps 1000 -model model
```

## Contribution

* If you find bug or want to new functions, Please write issue.
* If you fix your self, please fork and send pull request.

## LICENSE

This software license under [MIT](https://github.com/masato-ka/airc-rl-agent/blob/master/LICENCE) licence.

## Author

[masato-ka](https://github.com/masato-ka)
