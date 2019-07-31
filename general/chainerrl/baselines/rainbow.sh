# Treechop
python3 dqn_family.py \
  --gpu 0 --env MineRLTreechop-v0 --outdir results/MineRLTreechop-v0/rainbow \
  --noisy-net-sigma 0.5 --arch distributed_dueling --replay-capacity 300000 --replay-start-size 5000 \
  --target-update-interval 10000 --num-step-return 10 --agent CategoricalDoubleDQN --monitor --lr 0.0000625 \
  --adam-eps 0.00015 --prioritized --frame-stack 4 --frame-skip 4 --gamma 0.99 --batch-accumulator mean \
  --always-keys attack --reverse-keys forward --exclude-keys back left right sneak sprint

# # Navigate
# python3 dqn_family.py \
#   --gpu 0 --env MineRLNavigate-v0 --outdir results/MineRLNavigate-v0/rainbow \
#   --noisy-net-sigma 0.5 --arch distributed_dueling --replay-capacity 300000 --replay-start-size 5000 \
#   --target-update-interval 10000 --num-step-return 10 --agent CategoricalDoubleDQN --monitor --lr 0.0000625 \
#   --adam-eps 0.00015 --prioritized --frame-stack 4 --frame-skip 4 --gamma 0.99 --batch-accumulator mean \
#   --always-keys forward sprint attack --exclude-keys back left right sneak place

# # NavigateDense
# python3 dqn_family.py \
#   --gpu 0 --env MineRLNavigateDense-v0 --outdir results/MineRLNavigateDense-v0/rainbow \
#   --noisy-net-sigma 0.5 --arch distributed_dueling --replay-capacity 300000 --replay-start-size 5000 \
#   --target-update-interval 10000 --num-step-return 10 --agent CategoricalDoubleDQN --monitor --lr 0.0000625 \
#   --adam-eps 0.00015 --prioritized --frame-stack 4 --frame-skip 4 --gamma 0.99 --batch-accumulator mean \
#   --always-keys forward sprint attack --exclude-keys back left right sneak place

# # ObtainDiamond
# python3 dqn_family.py \
#   --gpu 0 --env MineRLObtainDiamond-v0 --outdir results/MineRLObtainDiamond-v0/rainbow \
#   --noisy-net-sigma 0.5 --arch distributed_dueling --replay-capacity 300000 --replay-start-size 5000 \
#   --target-update-interval 10000 --num-step-return 10 --agent CategoricalDoubleDQN --monitor --lr 0.0000625 \
#   --adam-eps 0.00015 --prioritized --frame-stack 4 --frame-skip 4 --gamma 0.99 --batch-accumulator mean \
#   --disable-action-prior
