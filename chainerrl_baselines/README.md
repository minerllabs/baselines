# MineRL Competition's baseline implementation with ChainerRL

Starter kit for [MineRL](https://github.com/minerllabs/minerl)
Competition with [ChainerRL](https://github.com/chainer/chainerrl).

# Resources

- [MineRL](https://github.com/minerllabs/minerl)
  - [Competition page](https://www.aicrowd.com/challenges/neurips-2019-minerl-competition)
  - [docs](http://minerl.io/docs/)
- [ChainerRL](https://github.com/chainer/chainerrl)
- [Chainer](https://chainer.org/)

# Installation

```sh
git clone <URL-for-this-repo>  # FIXME
cd minerl-chainerrl
pip install -r requirements.txt
```

See [MineRL installation](https://github.com/minerllabs/minerl#installation) and
[ChainerRL installation](https://github.com/chainer/chainerrl#installation) for more information.

# Getting started

- [baselines/path-to-double-dueling-dqn-script.sh]  # FIXME
    - Double Dueling DQN (DDDQN)
- [baselines/path-to-rainbow-script.sh]  # FIXME
    - Rainbow 
- [baselines/path-to-ppo-script.sh]  # FIXME
    - PPO

# Experimental results of DDDQN/Rainbow/PPO

Charts below show the *training* reward curves for each algorithm with prior knowledge of action/observation space.

Hyper parameters were chosen with grid search.

## MineRLTreechop-v0

Figure will be here


## MineRLNavigateDense-v0

Figure will be here


## MineRLNavigate-v0

Figure will be here


## MineRLObtainDiamond-v0

Figure will be here


## Prior knowledge for action/observation spaces

On `MineRLTreechop-v0`, `MineRLNavigateDense-v0` and `MineRLNavigate-v0`, Rainbow/PPO/DDDQN shape the action/observation space based on prior knowledge.  
The original idea of this prior knowledge came from [MineRL competition proposal paper](https://arxiv.org/abs/1904.10079)'s implementation
([Treechop](https://github.com/minerllabs/minerl/blob/master/tests/excluded/treechop_dqn_test.py),
[Navigate](https://github.com/minerllabs/minerl/blob/master/tests/excluded/navigate_dqn_test.py)).

The action spaces for MineRL environments are defined as OpenAI Gym's `Dict` space.
The set of space keys is different among tasks, but some of them are common
(namely, `forward`, `back`, `left`, `right`, `jump`, `sneak`, `sprint`, `attack` and `camera`).

`env_wrappers.SerialDiscreteActionWrapper` is the corresponding code for shaping the action space.

### Discretizing

The only action key which is continuous is `camera`.
`camera` is discretized into two-kinds action (PPO does not require discrete action space though):

```python
[(0, -10), (0, 10)]
```

### Disabling

Some of actions are disabled based on the task's characteristic.

There are two types of disabling.
`--always-keys` specifies actions which is always triggered throughout interaction with the environment.
These actions are removed from agent's action choice.

Actions specified as `--exclude-keys` are simply disabled and they will be never triggered.

On `MineRLTreechop-v0`, `--always-keys` is `attack` and `--exclude-keys` are `back`, `left`, `right`, `sneak`, `sprint`.  
On `MineRLNavigate-v0` / `MineRLNavigateDense-v0`, `--always-keys` are `forward`, `sprint`, `attack`
and `--exclude-keys` are `back`, `left`, `right`, `sneak` and `place`.

### Serializing

After discretizing and disabling, `Dict` action space is flattened and converted into one `Discrete` action space.  
And the resulting space is "serialized", that is, agents can choose only one of the action on the flattened action space
(the agent can push only one button of the gamepad in same time).

### (Reversing)

On Treechop, `forward` key is reversed. (`--reverse-keys forward`)  
Reversed keys are similar to the `--always-keys` actions, but they are not removed from agent's action choice.
Corresponding gamepad buttons for reversed actions are always pushed, but agent can choose to trigger the button off as one of the action.

### Exclusive actions

For Obtain* tasks, we employ "weak" action prior knowledge instead of prior knowledge described above.
It does not have `--always-keys`/`--exclude-keys`/`--reverse-keys` option,
but "exclusive" (or, "conflicting") actions are merged.

`forward` and `back` actions are exclusive, since they conflicts each other and pushing them at same time makes no sense.
They are merged and renamed as `forward_back` action with Discrete(3).  
(Original: forward 0/1, back 0/1. Merged: noop/forward/back)

List of exclusive actions we used:
  - `forward` / `back`
  - `right` / `left`
  - `sneak` / `sprint`
  - `attack` / `place` / `equip` / `craft` / `nearbyCraft` / `nearbySmelt`

See `env_wrappers.CombineActionWrapper` for more detail.

### Summary

Resulting action spaces after shaped with prior knowledge are:

- Treechop: Discrete(5)
- Navigate/NavigateDense: Discrete(6)
- Obtain*: Discrete(36)
