# MineRL Baselines

**`Towards Solving AI in Open World Environments`**

![viewer|64x64](http://www.minerl.io/docs/_images/survival1.mp4.gif)
![viewer|64x64](http://www.minerl.io/docs/_images/survival2.mp4.gif)
![viewer|64x64](http://www.minerl.io/docs/_images/survival3.mp4.gif)
![viewer|64x64](http://www.minerl.io/docs/_images/survival4.mp4.gif)


[![PyPI version](https://badge.fury.io/py/minerl.svg)](https://badge.fury.io/py/minerl)
[![Downloads](https://pepy.tech/badge/minerl)](https://pepy.tech/project/minerl)
[![Discord](https://img.shields.io/discord/565639094860775436.svg?label=&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2)](https://discord.gg/BT9uegr)


This repository contains baselines for various environments in the [`minerl` python package](https://github.com/minerllabs/minerl) as well as baseline submissions for the [MineRL Competition on Sample Efficicent RL @ NeurIPS 2019](https://www.aicrowd.com/organizers/minerl/challenges/neurips-2019-minerl-competition/). 


## Using the Baselines
The repository is broken up into two main folders:
```
competition/ # Baseline submissions for NeurIPS comp (MineRLObtainDiamond-v0)
   random_agent/ # git submodule, forks minerllabs/aicrowd_submission_template
   dqn_baseline/ # git submodule, forks ^^
   [...]

general/ # General baselines for the 6+ `minerl` environments!
   chainerrl/ # Baselines written in the Chainer RL framework
   [...]

```

To get started, do the following:
1.  Install the `minerl` python package: http://www.minerl.io/docs/tutorials/index.html
    - [Install JDK 8](http://www.minerl.io/docs/tutorials/index.html)
    - Install the MineRL Pacakge
        ```
        pip3 install --upgrade minerl
        ``` 
2. Clone the baselines **recursively**:
    ```
    git clone https://github.com/minerllabs/baselines.git --recurse-submodules 
    ```
3. Check out the baselines in [competition/](competition/) and [general/](general/)!

That's all! :) 

### Resources

* [**Documentation**](http://minerl.io/docs/) for the dataset and the environments is [found here!](http://minerl.io/docs/)
* [**Questions**](https://discourse.aicrowd.com/c/neurips-2019-minerl-competition) about getting started or running these baselines should be directed to the [competition forum](https://discourse.aicrowd.com/c/neurips-2019-minerl-competition) or [discord server](https://discord.gg/BT9uegr)
* [**Technical issues**](https://github.com/minerllabs/minerl/issues) related to the `minerl` python package should be submitted through the [MineRL GitHub page](https://github.com/minerllabs/minerl/issues)! 