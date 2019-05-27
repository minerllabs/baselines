# MineRL Competition
**`Sample Efficient Reinforcement Learning Through Human Demonstrations`**

This quick-start kit provides all of the resources needed to be successful in the MineRL Diamond Challenge. 


* **Submission, competition updates, and leaderboards** are available via the competition homepage: `https://www.aicrowd.com/challenges/neurips-2019-minerl-competition`
* **Questions** about getting started or rules of the competition should be directed to the competition forum: `https://discourse.aicrowd.com/c/neurips-2019-minerl-competition`  
* **Technical issues** related to this quick-start kit or python package should be submitted through the GitHub pages: `https://github.com/minenetproject/quick_start/issues` and `https://github.com/minenetproject/minerl/issues` respectively


## Environment
The MineRL Competiton uses a custom distribution of Microsoft's Malmo Env. This environment is packaged in the `minerl` package available via PyPI.

### Installation
`pip3 install minerl --user`

### Useage
`minerl` uses OpenAI gym wrapers providing the following environments:
* `MineRLTreechop-v0`
* `MineRLNavigate-v0`
* `MineRLNavigateDense-v0`
* `MineRLNavigateExtreme-v0`
* `MineRLNavigateExtremeDense-v0`
* `MineRLObtainIronPickaxe-v0`
* `MineRLObtainDiamond-v0`


## Data
The MineRL Competition leverages a large-scale dataset of human demonstrations - MineRLv0. To ensure access during evaluation, a python api is provided to load, filter, and map these demonstrations.
