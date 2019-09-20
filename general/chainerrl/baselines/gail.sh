# PLEASE MAKE SURE THAT DATABASE IS SET CORRECTLY.
# By default, this code assume that expert dataset is in ./expert_dataset
# Treechop
python3 gail.py \
  --gpu 0 --env MineRLTreechop-v0 --outdir results/MineRLTreechop-v0/gail \
  --policy ppo --frame-skip 8 --frame-stack 1 --num-camera-discretize 7 \
  --expert ./expert_dataset --action-wrapper multi-dimensional-softmax \
  --gamma 0.95 --gray-scale --policy-entropy-coef 0.005 \
  --discriminator-entropy-coef 0.005 --policy-update-interval 2000 \
  --policy-minibatch-size 200 --original-reward-weight 10 \
  --discriminator-update-interval 6000 --discriminator-minibatch-size 600

# # Navigate
# python3 gail.py \
#   --gpu 0 --env MineRLNavigate-v0 --outdir results/MineRLNavigate-v0/gail \
#   --policy ppo --frame-skip 8 --frame-stack 1 --num-camera-discretize 7 \
#   --expert ./expert_dataset --action-wrapper multi-dimensional-softmax \
#   --gamma 0.95 --gray-scale --policy-entropy-coef 0.005 \
#   --discriminator-entropy-coef 0.005 --policy-update-interval 2000 \
#   --policy-minibatch-size 200 --original-reward-weight 10 \
#   --discriminator-update-interval 6000 --discriminator-minibatch-size 600

# # NavigateDense
# python3 gail.py \
#   --gpu 0 --env MineRLNavigateDense-v0 --outdir results/MineRLNavigateDense-v0/gail \
#   --policy ppo --frame-skip 8 --frame-stack 1 --num-camera-discretize 7 \
#   --expert ./expert_dataset --action-wrapper multi-dimensional-softmax \
#   --gamma 0.95 --gray-scale --policy-entropy-coef 0.005 \
#   --discriminator-entropy-coef 0.005 --policy-update-interval 2000 \
#   --policy-minibatch-size 200 --original-reward-weight 10 \
#   --discriminator-update-interval 6000 --discriminator-minibatch-size 600

# # ObtainDiamond
# python3 gail.py \
#   --gpu 0 --env MineRLObtainDiamond-v0 --outdir results/MineRLObtainDiamond-v0/gail \
#   --policy ppo --frame-skip 8 --frame-stack 1 --num-camera-discretize 11 \
#   --expert ./expert_dataset --action-wrapper multi-dimensional-softmax \
#   --gamma 0.95 --gray-scale --policy-entropy-coef 0.01 \
#   --discriminator-entropy-coef 0.1 --policy-update-interval 2000 \
#   --policy-minibatch-size 200 --original-reward-weight 10 \
#   --discriminator-update-interval 6000 --discriminator-minibatch-size 600
