#!/bin/bash


echo "** Install requirement"
sudo apt update
sudo apt install -y liblapack-dev python3-scipy libfreetype6-dev
sudo pip3 install Cython gym git+https://github.com/tawnkramer/gym-donkeycar.git#egg=gym-donkeycar
sudo pip3 install git+https://github.com/masato-ka/stable-baselines.git@v2.9.0-jetson#egg=stable-baselines

echo "** Building..."
sudo pip3 install .\[jetpack]\

echo "** Install learning_racer successfully"
echo "** Bye :)"