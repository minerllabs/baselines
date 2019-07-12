# MineRL 2019 Competition @ NeurIPS: Quick Start
**`Sample Efficient Reinforcement Learning Through Human Demonstrations`**


[![Support us on patron](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fshieldsio-patreon.herokuapp.com%2Fwguss_imushroom&style=for-the-badge)](https://www.patreon.com/wguss_imushroom)
[![Downloads](https://pepy.tech/badge/minerl)](https://pepy.tech/project/minerl)
[![Discord](https://img.shields.io/discord/565639094860775436.svg?label=&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2)](https://discord.gg/BT9uegr)

This quick-start kit provides all of the resources needed to be successful in the MineRL Diamond Challenge. 

** [The full documentation for the dataset and the environments is found here!](http://minerl.io/docs/)

* **Submission, competition updates, and leaderboards** are available via the [competition homepage](https://www.aicrowd.com/challenges/neurips-2019-minerl-competition) 
  > note that submissions and leaderboards are not availabe yet
* **Questions** about getting started or rules of the competition should be directed to the [competition forum](https://discourse.aicrowd.com/c/neurips-2019-minerl-competition) or [discord server](https://discord.gg/BT9uegr)
* **Technical issues** related to the code should be submitted through the [MineRL GitHub page](https://github.com/minerllabs/minerl/issues) This repo may not be monitored!


## Environment
The MineRL Competiton uses a custom distribution of Microsoft's Malmo Env. This environment is packaged in the `minerl` package available via PyPI. The documentation can be found [here](http://minerl.io/docs/)

### Installation
Ensure JDK-8 is installed and then simply (python3.5+)
`pip3 install minerl --user`

**For a full guide please checkout the guide**

### Environments
`minerl` uses OpenAI gym wrappers for the following environments with accompanying data:
* `MineRLTreechop-v0`
* `MineRLNavigate-v0`
* `MineRLNavigateDense-v0`
* `MineRLNavigateExtreme-v0`
* `MineRLNavigateExtremeDense-v0`
* `MineRLObtainIronPickaxe-v0`
* `MineRLObtainIronPickaxeDense-v0`
* **`MineRLObtainDiamond-v0`**
| All agents will be evaluated on this environment
* `MineRLObtainDiamondDense-v0`

`minerl` also currently includes a few debug environments for testing that lack any data:
* `MineRLNavigateDenseFixed-v0`
* `MineRLObtainTest-v0`

## Data

The MineRL Competition leverages a large-scale dataset of human demonstrations - MineRLv0. To ensure access during evaluation, a python api is provided to load demonstrations. Currently the data is almost 15GB, ensure ample space before downloading! [To see how check out the quick start guide!](http://minerl.io/docs/tutorials/data_sampling.html)

![viewer|540x272](http://www.minerl.io/docs/_images/cropped_viewer.gif)

## Baselines
To get started with some baselines check out the `chainerrl_baselines/` folder in this repository!

## Submitting
** Submissions are coming soon! **


