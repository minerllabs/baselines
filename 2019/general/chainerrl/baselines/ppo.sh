# Treechop
python3 ppo.py \
  --gpu 0 --env MineRLTreechop-v0 --outdir results/MineRLTreechop-v0/ppo \
  --arch nature --update-interval 1024 --monitor --lr 0.00025 --frame-stack 4 --frame-skip 4 --gamma 0.99 --epochs 3 \
  --always-keys attack --reverse-keys forward --exclude-keys back left right sneak sprint

# # Navigate
# python3 ppo.py \
#   --gpu 0 --env MineRLNavigate-v0 --outdir results/MineRLNavigate-v0/ppo \
#   --arch nature --update-interval 1024 --monitor --lr 0.00025 --frame-stack 4 --frame-skip 4 --gamma 0.99 --epochs 3 \
#   --always-keys forward sprint attack --exclude-keys back left right sneak place

# # NavigateDense
# python3 ppo.py \
#   --gpu 0 --env MineRLNavigateDense-v0 --outdir results/MineRLNavigateDense-v0/ppo \
#   --arch nature --update-interval 1024 --monitor --lr 0.00025 --frame-stack 4 --frame-skip 4 --gamma 0.99 --epochs 3 \
#   --always-keys forward sprint attack --exclude-keys back left right sneak place

# # ObtainDiamond
# python3 ppo.py \
#   --gpu 0 --env MineRLObtainDiamond-v0 --outdir results/MineRLObtainDiamond-v0/20190701_v17/check_ppo \
#   --arch nature --update-interval 1024 --monitor --lr 0.00025 --frame-stack 4 --frame-skip 4 --gamma 0.99 --epochs 3 \
#   --disable-action-prior
