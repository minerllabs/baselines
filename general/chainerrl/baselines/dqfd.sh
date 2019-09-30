# Treechop
python3 train_dqfd.py \
  --env MineRLTreechop-v0 --expert-demo-path ./expert_dataset/MineRLTreechop-v0/ \
  --frame-skip 4 --frame-stack 4 --gpu 0 --lr 6.25e-5 --minibatch-size 32 \
  --n-experts 16 --use-noisy-net before-pretraining

# Navigate
#python3 train_dqfd.py \
#  --env MineRLNavigate-v0 --expert-demo-path ./expert_dataset/MineRLNavigate-v0/ \
#  --frame-skip 4 --frame-stack 4 --gpu 0 --lr 6.25e-5 --minibatch-size 32 \
#  --n-experts 20 --use-noisy-net before-pretraining --use-full-observation

# Diamond
#python3 train_dqfd.py \
#  --env MineRLObtainDiamond-v0 --expert-demo-path ./expert_dataset/MineRLObtainDiamond-v0/ \
#  --frame-skip 4 --frame-stack 4 --gpu 0 --lr 6.25e-5 --minibatch-size 32 \
#  --n-experts 20 --use-noisy-net before-pretraining --use-full-observation

# DiamondDense
#python3 train_dqfd.py \
#  --env MineRLObtainDiamondDense-v0 --expert-demo-path ./expert_dataset/MineRLObtainDiamondDense-v0/ \
#  --frame-skip 4 --frame-stack 4 --gpu 0 --lr 6.25e-5 --minibatch-size 32 \
#  --n-experts 20 --use-noisy-net before-pretraining --use-full-observation