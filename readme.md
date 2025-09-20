# AdaptFlip
Repository for code on denoising approaches in recommendation systems

For Running AdaptFlip Code for NCF, go to `AdaptFlip` directory and run:
```bash
python -u main.py --dataset 'movielens' --W 3 --alpha 1.5
```
or alternatively, modify the `run.sh` file and run it directly.

And for running for CDAE, run:
```bash
python -u cdae.py --dataset 'movielens' --W 3 --alpha 1.5
```
or alternatively, modify the `cdae.sh` file and run it directly.

All the logs will be formed for the shell files will be formed in the `logs/<dataset>` folder.