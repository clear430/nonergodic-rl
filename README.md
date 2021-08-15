# Revisiting Fundamentals of Model-Free Reinforcement Learning

Implementation using Python 3.9.6 and PyTorch 1.9.0 with CUDA 11.1. 

Based entirely on extending and correcting the previously submitted project in [r-grewal/data5709](https://github.com/r-grewal/data5709).

Code tested on both Windows 10 21H1 and Ubuntu 20.04.2.0 LTS using an AMD Ryzen 7 5800X, Nvidia RTX 3070, 64GB 3600MHz CL16 RAM, and a Samsung SSD 980 Pro. Executable on the CentOS 6.9 Linux-based Artemis HPC after manually installing or updating several packages (gym, PyBullet, PyTorch, etc.) using source wheel files.

The research encompasses several overlapping areas: 
1. Peculiarities regarding use of different critic loss functions, tail exponents, and shadow means
2. Multi-step returns and replay buffer coupling in continuous action spaces, and tail exponents
3. Multiplicative (or non-ergodic) reinforcement learning, with insights from the previous two areas
4. Designing energy efficient multi-stage actors for operation in extremely remote environments

## References
* Reinforcement learning ([Szepesvári 2009](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf), [Sutton and Bartow 2018](http://incompleteideas.net/book/RLbook2020.pdf))
* Feature reinforcement learning ([Hutter 2009](https://sciendo.com/downloadpdf/journals/jagi/1/1/article-p3.pdf), [Hutter 2016](https://www.sciencedirect.com/science/article/pii/S0304397516303772), [Majeed and Hutter 2018](https://www.ijcai.org/Proceedings/2018/0353.pdf))
* Twin Delayed DDPG (TD3) ([Silver et al. 2014](http://proceedings.mlr.press/v32/silver14.pdf), [Lillicrap et al. 2016](https://arxiv.org/pdf/1509.02971.pdf), [Fujimoto et al. 2018](https://arxiv.org/pdf/1802.09477.pdf))
* Soft Actor-Critic (SAC) ([Ziebart 2010](https://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf), [Haarnoja et al. 2017](http://proceedings.mlr.press/v70/haarnoja17a/haarnoja17a-supp.pdf), [Haarnoja et al. 2018](https://arxiv.org/pdf/1812.05905.pdf))
* Critic loss functions ([Guan et al. 2019](https://tongliang-liu.github.io/papers/TPAMITruncatedNMF.pdf))
* Non-i.i.d. and fat-tailed data ([Fazekas and Klesov 2006](https://epubs.siam.org/doi/pdf/10.1137/S0040585X97978385), [Cirillo and Taleb 2016](https://www.tandfonline.com/doi/pdf/10.1080/14697688.2016.1162908?needAccess=true), [Cirillo and Taleb 2020](https://www.nature.com/articles/s41567-020-0921-x.pdf), [Taleb 2020](https://arxiv.org/ftp/arxiv/papers/2001/2001.10488.pdf))
* Coupling of multi-step returns and experience replay ([Fedus et al. 2020](https://arxiv.org/pdf/2007.06700.pdf))
* Primer on statistical mechanics, ensemble averages, and entropy ([Landau and Lifshitz 1969](https://archive.org/details/ost-physics-landaulifshitz-statisticalphysics))
* Multiplicative dynamics ([Peters 2011a](https://www.tandfonline.com/doi/pdf/10.1080/14697688.2010.513338?needAccess=true), [Peters 2011b](https://royalsocietypublishing.org/doi/pdf/10.1098/rsta.2011.0065), [Peters and Gell-Mann 2016](https://aip.scitation.org/doi/pdf/10.1063/1.4940236), [Peters 2019](https://www.nature.com/articles/s41567-019-0732-0.pdf), [Peters et al. 2020](https://arxiv.org/ftp/arxiv/papers/2005/2005.00056.pdf), [Peters and Adamou 2020](https://arxiv.org/pdf/1101.4548.pdf), [Meder et al. 2020](https://arxiv.org/ftp/arxiv/papers/1906/1906.04652.pdf), [Peters and Adamou 2021](https://arxiv.org/pdf/1801.03680.pdf), [Spitznagel 2021](https://www.wiley.com/en-us/Safe+Haven%3A+Investing+for+Financial+Storms-p-9781119401797))
* Power consumption of neural networks ([Han et al. 2015](https://proceedings.neurips.cc/paper/2015/file/ae0eb3eed39d2bcef4622b2499a05fe6-Paper.pdf), [García-Martín et al. 2019](https://www.sciencedirect.com/science/article/pii/S0743731518308773))

## Acknowledgements
The Sydney Informatics Hub and the University of Sydney’s high performance computing cluster, Artemis, for providing the computing resources that have contributed to the results reported herein.

The base TD3 and SAC algorithms were implemented using guidance from the following repositories.
* [DLR-RM/stable-baelines3](https://github.com/DLR-RM/stable-baselines3)
* [haarnoja/sac](https://github.com/haarnoja/sac)
* [openai/spinningup](https://github.com/openai/spinningup)
* [p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch)
* [philtabor/Actor-Critic-Methods-Paper-To-Code](https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code)
* [rail-berkley/softlearning](https://github.com/rail-berkeley/softlearning) 
* [rlworkgroup/garage](https://github.com/rlworkgroup/garage)
* [sfujim/TD3](https://github.com/sfujim/TD3/)
