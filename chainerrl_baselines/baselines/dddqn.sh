# Treechop
python3 dqn_family.py \
  --gpu 0 --env MineRLTreechop-v0 --outdir results/MineRLTreechop-v0/dddqn \
  --final-exploration-frames 1000000 --final-epsilon 0.02 --arch dueling --replay-capacity 800000 --replay-start-size 1000 \
  --target-update-interval 1000 --update-interval 1 --agent DoubleDQN --monitor --lr 0.001 --gamma 0.99 --batch-accumulator mean \
  --always-keys attack --reverse-keys forward --exclude-keys back left right sneak sprint

# # Navigate
# python3 dqn_family.py \
#   --gpu 0 --env MineRLNavigate-v0 --outdir results/MineRLNavigate-v0/dddqn \
#   --final-exploration-frames 100000 --final-epsilon 0.02 --arch dueling --replay-capacity 30000 --replay-start-size 1000 \
#   --target-update-interval 1000 --update-interval 1 --agent DoubleDQN --monitor --lr 0.0005 --gamma 1.0 --batch-accumulator mean \
#   --always-keys forward sprint attack --exclude-keys back left right sneak place

# # NavigateDense
# python3 dqn_family.py \
#   --gpu 0 --env MineRLNavigateDense-v0 --outdir results/MineRLNavigateDense-v0/dddqn \
#   --final-exploration-frames 100000 --final-epsilon 0.02 --arch dueling --replay-capacity 30000 --replay-start-size 1000 \
#   --target-update-interval 1000 --update-interval 1 --agent DoubleDQN --monitor --lr 0.0005 --gamma 1.0 --batch-accumulator mean \
#   --always-keys forward sprint attack --exclude-keys back left right sneak place
