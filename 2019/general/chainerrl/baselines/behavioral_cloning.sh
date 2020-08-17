# PLEASE MAKE SURE THAT DATABASE IS SET CORRECTLY.
# By default, this code assume that expert dataset is in ./expert_dataset
# Treechop
python3 behavioral_cloning.py \
 --gpu 0 --env MineRLTreechop-v0 --outdir results/MineRLTreechop-v0/behavioral_cloning \
 --frame-skip 8 --frame-stack 1 --always-keys attack --reverse-keys forward \
 --exclude-keys back left right sneak sprint --num-camera-discretize 7 \
 --expert ./expert_dataset \
 --action-wrapper discrete --gray-scale \
 --entropy-coef 0.001

# # Navigate
# python3 behavioral_cloning.py \
#   --gpu 0 --env MineRLNavigate-v0 --outdir results/MineRLNavigate-v0/behavioral_cloning \
#   --frame-skip 8 --frame-stack 1 --always-keys sprint attack forward \
#   --exclude-keys back left right sneak place --prioritized-elements camera \
#   --num-camera-discretize 7 --expert ./expert_dataset \
#   --action-wrapper discrete --gray-scale --entropy-coef 0.001

# # NavigateDense
# python3 behavioral_cloning.py \
#   --gpu 0 --env MineRLNavigateDense-v0 --outdir results/MineRLNavigateDense-v0/behavioral_cloning \
#   --frame-skip 8 --frame-stack 1 --always-keys sprint attack forward \
#   --exclude-keys back left right sneak place --prioritized-elements camera \
#   --num-camera-discretize 7 --expert ./expert_dataset \
#   --action-wrapper discrete --gray-scale --entropy-coef 0.001

# # ObtainDiamond
# python3 behavioral_cloning.py \
#   --gpu 0 --env MineRLObtainDiamond-v0 --outdir results/MineRLObtainDiamond-v0/behavioral_cloning \
#   --frame-skip 8 --frame-stack 1 --num-camera-discretize 11 \
#   --expert ./expert_dataset --action-wrapper multi-dimensional-softmax \
#   --gray-scale --entropy-coef 0.001
