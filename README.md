AI RC Car RL Agent
===

Overview

This software is able to Self learning your AI RC Car 
by Deep reinforcement learning in few min.


## Description

AI RC Car like JetBot or JetRacer, DonkeyCar are  learning by supervised-learning.
supervised-learning needs much labeled data that written human. The behaivior quality 
is determined that data. Running behavior characteristic is determined that data.

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
JetBot is learning running behavior on road in under 30 min. Software is running on Jetson Nano.  

[![](https://img.youtube.com/vi/j8rSWvcO-s4/0.jpg)](https://www.youtube.com/watch?v=j8rSWvcO-s4)


## Setup

### Requirements

* JetBot or JetRacer base image(Recommend latest images)
* tensorflow-gpu
* torch
* torchvision
* OpenCV

### Install

Dependency library install.

* OpenAIGym 0.10.9

```
sudo pip3 install gym==0.10.9
```

* stable-baselines v2.9.0

```
$ cd ~/ && git clone https://github.com/hill-a/stable-baselines.git -b v2.9.0
$ cd stable-baselines/ && sudo python3 setup.py install
```

* clone this repository

```
$ cd ~/ && git clone https://github.com/masato-ka/airc-rl-agent.git
$ cd airc-rl-agent
```



## Usage

## Contribution

* If you find bug or want to new functions, Please write issue.
* If you fix your self, please fork and send pull request.

## Licence

This software license under [MIT](https://github.com/masato-ka/airc-rl-agent/blob/master/LICENCE) licence.

## Author

[masato-ka](https://github.com/masato-ka)
