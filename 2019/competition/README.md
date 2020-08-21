# MineRL 2019 Competition @ NeurIPS: Baseline Submissions
**`Sample Efficient Reinforcement Learning Through Human Demonstrations`**

[![PyPI version](https://badge.fury.io/py/minerl.svg)](https://badge.fury.io/py/minerl)
[![Downloads](https://pepy.tech/badge/minerl)](https://pepy.tech/project/minerl)
[![Discord](https://img.shields.io/discord/565639094860775436.svg?label=&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2)](https://discord.gg/BT9uegr)

This folder contains a set of baselines for solving the `MineRLObtainDiamond-v0` environment in the NeurIPS 2019 MineRL Competition. 

* [**Main Competition**](https://www.aicrowd.com/challenges/neurips-2019-minerl-competition) where you can *sign-up*, read the rules, and check the leaderboard is [here](https://www.aicrowd.com/challenges/neurips-2019-minerl-competition).
* [**Submission Starter Template**](https://github.com/minerllabs/competition_submission_starter_template) with documentation and starter code for your first submission to the competition. (The baselines in this repository are based off of this!)
* [**Documentation**](http://minerl.io/docs/) for the dataset and the environments is [found here!](http://minerl.io/docs/)
* [**Questions**](https://discourse.aicrowd.com/c/neurips-2019-minerl-competition) about getting started or running these baselines should be directed to the [competition forum](https://discourse.aicrowd.com/c/neurips-2019-minerl-competition) or [discord server](https://discord.gg/BT9uegr)
* [**Technical issues**](https://github.com/minerllabs/minerl/issues) related to the `minerl` python package should be submitted through the [MineRL GitHub page](https://github.com/minerllabs/minerl/issues)! 


![viewer|540x272](http://www.minerl.io/docs/_images/cropped_viewer.gif)

## Getting Started w/ the Baseline Submissions


**Prerequisites.** To get started with these baselines and incorporate them into your own submissions make sure you have:
1. [Registered with the competition](https://www.aicrowd.com/challenges/neurips-2019-minerl-competition) on the main competition page,
2. [Cloned your own submission starter template](https://github.com/minerllabs/competition_submission_starter_template) and [created a repository at AIcrowd](http://gitlab.aicrowd.com).

**Usage.** The various baselines associated with this competition are contained within this folder (`competition/`) as `git` repositories (submodules) which fork the [AIcrowd submission template](https://github.com/minerllabs/competition_submission_starter_template). 

To download all of the baseline code run the following:
```
git clone https://github.com/minerllabs/baselines.git --recurse-submodules
```
Each competition baseline is structured exactly as a submission to AIcrowd (containing `train.py`, `test.py`, etc.)  and contains a link to the its leaderboard score!

You can additionally base your submission off of one of the existing baselines in this repository by simply forking or cloning one of the baseline repositories in this folder! Good luck :)
