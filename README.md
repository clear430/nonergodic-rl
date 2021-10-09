# Revisiting Fundamentals of Model-Free Reinforcement Learning

Research encompasses several overlapping areas: 
1. Peculiarities regarding use of critic loss functions, tail exponents, and shadow means,
2. Multi-step returns and replay buffer coupling in continuous action spaces,
3. Reinforcement learning in multiplicative (non-ergodic) domains, maximising the time-average growth rate, and
4. Designing energy efficient multi-stage actors for operation in extremely remote environments.

Implementation using [Python](https://www.python.org) 3.9.7 and [PyTorch](https://pytorch.org) 1.9.1 with [CUDA](https://developer.nvidia.com/cuda-zone) 11.1. 
- Built using an AMD Ryzen 7 5800X, Nvidia RTX 3070, 64GB RAM, and a Samsung SSD 980 Pro.
- Tested on [Pop!\_OS](https://pop.system76.com) 21.04, [Ubuntu](https://ubuntu.com) 20.04 LTS, [Arch](https://archlinux.org) 2021.10.01, and Windows 10/11 21H2.
- Experiments performed on the [Artemis](https://sydneyuni.atlassian.net/wiki/spaces/RC/pages/1033929078/Artemis+HPC+documentation) high performance computing cluster using [CentOS](https://www.centos.org) 6.9.

Research based on extending a [capstone](https://github.com/rgrewa1/capstone) project submitted in June 2021 at the [University of Sydney](https://www.sydney.edu.au), Australia.

## Key Findings
**Additive Dynamics**
* Critic loss aggregation using MSE is an acceptable starting point but use of HUB, MAE, and HSC should be considered as there exists a string potential for ‘free’ performance gains dependent on environment.
* Critic loss mini-batches appear to exhibit extreme kurtosis (fat tails) and so aggregating them using an empirical arithmetic mean (Monte-Carlo approach) severely underestimates the true population mean.
* Multi-step returns for continuous action spaces using TD3 and SAC is not advised due to lack of global policy maximisation across the action space unlike the discrete case.

**Multiplicative Dynamics**
* The maximisation of probability-based expectations methodology offer by contemporary decision theory is wholly inappropriate for maximising wealth in multiplicative processes due to conflation of probabilities with payoffs.
*  State-of-the-art model-free reinforcement learning algorithms (TD3 and SAC) designed to maximise expected additive rewards are modified to operate in any multiplicative environment.  
* The model-free agent now fully autonomously, self-learns the actions required to maximise wealth through the avoidance of steep losses, represented by raising the time-average growth rate.
* The theory is experimentally validated by converging to known optimal growth-maximising actions (leverages) for gambles involving coin flips, die rolls, and geometric Brownian motion. 
* Cost-effective risk mitigation using extremely convex insurance safe havens is investigated where the model-free agent develops a strategy that indisputably increases wealth by reducing the amount of risk taken.
* Direct applications encompass any situation where percentage changes (as opposed to numerical changes) in underlying values are reported, such as financial trading, economic modelling, and guidance systems. 

## Data Analysis
Comprehensive discussion and implications of all results are described in `docs/RGrewal_RL.pdf`.

The data regarding agent training performance (NumPy arrays), the learned models (PyTorch parameters), and coin flip experiments (NumPy arrays) have a total combined size of 16.5 GB. 

The breakdown for additive agents, multiplicative agents, and coin flip experiments are 2.5 GB, 13.7 GB, and 80 MB respectively. All data is available upon request.  

## Usage 
Using the [release](https://github.com/rgrewa1/nonergodic-rl/releases) is recommended (contains only executable code). Contents of the `docs/` directory are excluded.

All reinforcement learning agent training is executed using `main.py` with instructions provided within the file. Upon the completion of each experiment, relevant directories within `models/` and `results/` titled by the environment name will be created containing all output data and summary plots. 

Final aggregated figures for all related experiments that share common training parameters are generated using `extras/gen_figures.py` and outputted in `docs/figs/`. The exact aggregation details must be inputted in the file.

Binary coin flip experiments pertaining to empirical optimal leverages are conducted using `scripts/exp_multiverse.py` with output data placed in `results/multiverse/` and  summary figures placed in `docs/figs/`. 

The general process for executing the code involves the following commands:
```commandline
git clone https://github.com/rgrewa1/nonergodic-rl.git

cd nonergodic-rl

python -m venv rl_env

source rl_env/bin/activate

pip3 install -r requirements.txt

python main.py

python scripts/exp_multiverse.py
```

## References
* Reinforcement learning ([Szepesvári 2009](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf), [Sutton and Bartow 2018](http://incompleteideas.net/book/RLbook2020.pdf))
* Feature reinforcement learning ([Hutter 2009](https://sciendo.com/downloadpdf/journals/jagi/1/1/article-p3.pdf), [Hutter 2016](https://www.sciencedirect.com/science/article/pii/S0304397516303772), [Majeed and Hutter 2018](https://www.ijcai.org/Proceedings/2018/0353.pdf))
* Twin Delayed DDPG (TD3) ([Silver et al. 2014](http://proceedings.mlr.press/v32/silver14.pdf), [Lillicrap et al. 2016](https://arxiv.org/pdf/1509.02971.pdf), [Fujimoto et al. 2018](https://arxiv.org/pdf/1802.09477.pdf))
* Soft Actor-Critic (SAC) ([Ziebart 2010](https://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf), [Haarnoja et al. 2017](http://proceedings.mlr.press/v70/haarnoja17a/haarnoja17a-supp.pdf), [Haarnoja et al. 2018](https://arxiv.org/pdf/1812.05905.pdf))
* Critic loss functions from NMF ([Guan et al. 2019](https://arxiv.org/pdf/1906.00495.pdf))
* Multi-step returns and replay coupling ([De Asis et al. 2018](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16294/16593), [Meng, Gorbet and Kulic 2020](https://arxiv.org/pdf/2006.12692.pdf), [Fedus et al. 2020](https://arxiv.org/pdf/2007.06700.pdf))
* Non-i.i.d. data and fat tails ([Fazekas and Klesov 2006](https://epubs.siam.org/doi/pdf/10.1137/S0040585X97978385), [Cirillo and Taleb 2016](https://www.tandfonline.com/doi/pdf/10.1080/14697688.2016.1162908?needAccess=true), [Cirillo and Taleb 2020](https://www.nature.com/articles/s41567-020-0921-x.pdf), [Taleb 2020](https://arxiv.org/ftp/arxiv/papers/2001/2001.10488.pdf))
* Primer on statistical mechanics, ensemble averages, and entropy ([Landau and Lifshitz 1969](https://archive.org/details/ost-physics-landaulifshitz-statisticalphysics))
* Multiplicative dynamics ([Bernoulli 1738](http://risk.garven.com/wp-content/uploads/2013/09/St.-Petersburg-Paradox-Paper.pdf), [Kelly 1956](https://cpb-us-w2.wpmucdn.com/u.osu.edu/dist/7/36891/files/2017/07/Kelly1956-1uwz47o.pdf), [Peters 2011a](https://www.tandfonline.com/doi/pdf/10.1080/14697688.2010.513338?needAccess=true), [Peters 2011b](https://royalsocietypublishing.org/doi/pdf/10.1098/rsta.2011.0065), [Peters and Gell-Mann 2016](https://aip.scitation.org/doi/pdf/10.1063/1.4940236), [Peters 2019](https://www.nature.com/articles/s41567-019-0732-0.pdf), [Peters et al. 2020](https://arxiv.org/ftp/arxiv/papers/2005/2005.00056.pdf), [Meder et al. 2020](https://arxiv.org/ftp/arxiv/papers/1906/1906.04652.pdf), [Peters and Adamou 2021](https://arxiv.org/pdf/1801.03680.pdf), [Spitznagel 2021](https://www.wiley.com/en-us/Safe+Haven%3A+Investing+for+Financial+Storms-p-9781119401797))
* Power consumption of neural networks ([Han et al. 2015](https://proceedings.neurips.cc/paper/2015/file/ae0eb3eed39d2bcef4622b2499a05fe6-Paper.pdf), [García-Martín et al. 2019](https://www.sciencedirect.com/science/article/pii/S0743731518308773))

## Acknowledgements
The [Sydney Informatics Hub](https://www.sydney.edu.au/research/facilities/sydney-informatics-hub.html) and the University of Sydney’s high performance computing cluster, Artemis, for providing the computing resources that have contributed to the results reported herein.

The base TD3 and SAC algorithms were implemented using guidance from the following repositories: [DLR-RM/stable-baelines3](https://github.com/DLR-RM/stable-baselines3), [haarnoja/sac](https://github.com/haarnoja/sac), [openai/spinningup](https://github.com/openai/spinningup), [p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch), [philtabor/Actor-Critic-Methods-Paper-To-Code](https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code), [rail-berkley/softlearning](https://github.com/rail-berkeley/softlearning), [rlworkgroup/garage](https://github.com/rlworkgroup/garage), and [sfujim/TD3](https://github.com/sfujim/TD3/).
