# MineRL Competition
**`Sample Efficient Reinforcement Learning Through Human Demonstrations`**

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
The MineRL Competition leverages a large-scale dataset of human demonstrations - MineRLv0. To ensure access during evaluation, a python api is provided to load demonstrations. Currently the data is almost 15GB, ensure ample space before downloading!

## Samples
There are 4 example scripts in sample.py:
* `python sample.py env` Test out a random agent on treechop
* `python sample.py data` View a sample of the dataset
* `python sample.py test` See a deterministic bot perform complex actions
* `python sample.py download` A simple command line utility for downloading the dataset
