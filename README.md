# TrackMania 2020 AI driver
*Hold on tight as we merge the high-octane world of car racing with the cutting-edge capabilities of Reinforcement Learning, creating an unstoppable force on the Trackmania race tracks*
...well, not as exciting as ChatGPT puts it, yet we found an idea of training RL algorithm on a real car racing game quite fascinating! We use a Steam version of Trackmania 2020 and [TMRL](https://github.com/trackmania-rl/tmrl) environment.

## How to reproduce the experiments?
1.  `pip install -r requirements.txt`
2.  Install TMRL environment on your [Windows](https://github.com/trackmania-rl/tmrl/blob/master/readme/Install.md) or [Linux](https://github.com/trackmania-rl/tmrl/blob/master/readme/install_linux.md) machine.
3.  Enter the directory of the experiment you want to reproduce.
4.  Copy or create a soft link to a config file (the destination is `C:/Users/username/TmrlData/config/config.json` or `/home/username/TmrlData/config/config.json` depending on your OS)

## Running an experiments
1. To run evaluation for ad-hoc algorithm, cd to `adhoc` and run `python run.py`. You will see a car driving in the track!
2. To train one of our custom models run 3 terminals and launch commands from experiment directory: `python -m custom_train.py --server`, `python -m custom_train.py --trainer`, and `python -m custom_train.py --worker`. *Watch all Star Wars movies while the network is training.... Probably twice:)*
