# set -eux

# DIR="/mnt/vol12/nakata/home/minerl2019/minerl-chainerrl/baselines/results/first_release"
DIR="/mnt/vol12/nakata/home/minerl2019/baselines/general/chainerrl/baselines/results/v23_20190906"

for file in $(find $DIR -name "log.txt"); do
  python convert_log.py --file $file
done
