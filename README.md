# Optimization of Join Order Sequence Pattern

The optimization of join order sequence pattern is a crucial step in the process of query optimization, which involves identifying the most efficient sequence of table joins to produce the desired results with minimum computational resources. The aim of this repository is to address the problem of join ordering optimization, which is still an active area of research, and to explore various techniques that have been developed to address this challenge.

In practice, a typical database system can execute an SQL query in multiple ways, depending on the order of selected operators and algorithms. Therefore, selecting the appropriate join order is an essential decision that can significantly affect the overall performance of the optimizer. The difference between optimal and non-optimal join orders can be orders of magnitude, highlighting the importance of selecting the proper order of joins.

To address this challenge, several techniques have been proposed, including ReJoin, DQ, SkinnerDB, and RTOS. These techniques employ various machine learning algorithms, such as deep reinforcement learning and tree-LSTM, to identify the most efficient join order sequence. While some studies employ complex structures, such as LSTM techniques, others use simpler structures.

- ReJoin: Deep Reinforcement Learning for Join Order Enumeration 
    * aiDM’18
    * 10 June 2018
    * Marcus, R., & Papaemmanouil, O.

- DQ: Learning to Optimize Join Series With Deep Reinforcement Learning
    * arXiv:1808.03196v2 [cs.DB]
    * 10 January 2019
    * Krishnan, S., Yang, Z., Goldberg, K., Hellerstein, J., & Stoica, I.

- SkinnerDB: Regret-Bounded Query Evaluation via Reinforcement Learning
    * arXiv:1901.05152v1 [cs.DB]
    * 16 January 2019
    * Trummer, I., Wang, J., Wei, Z., Maram, D., Moseley, S., Jo, S., ... & Rayabhari, A.

- RTOS: Reinforcement Learning with Tree-LSTM for Join Order Selection
    * 2020 IEEE 36th International Conference on Data Engineering (ICDE)
    * 20-24 April 2020
    * Yu, X., Li, G., Chai, C., & Tang, N.

This repository aims to study the join ordering problem and estimate the complexity of join planning by addressing each step in detail and one by one.

***

### Environment Setup

Spesification of environment:
- **System**: Windows 11 Pro 64-bit operating system, x64-based processor
- **Proccessor**: 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz   2.80 GHz
- **RAM**: 32.0 GB

### Python and Install PostgreSQL 
- You can download the programs from the link below.

[Download link for Python](https://www.python.org/downloads/)
[Download link for PostgreSQL](https://www.postgresql.org/download/windows/) 

```
Used Python Version: Python 3.7.9
Used PostgreSQL Version: postgresql-15.2-1-windows-x64
```

### Install All Libraries 
- Install all libraries from ".\requirements.txt" file by using pip install. 

```
pip install -r \path\to\requirements.txt
```

### Download all IMDB dataset
- Download the files from the following address and put all of them in the same directory (You can use file explorer.): 

**FTP Link**: ftp://ftp.funet.fi/pub/mirrors/ftp.imdb.com/pub/frozendata/

### Download and Install the cinemagoer library
- You can directly access to the repostory from below link or just install it via pip. 

[https://github.com/cinemagoer/cinemagoer](https://github.com/cinemagoer/cinemagoer) 


```
pip install cinemagoer
```

**Release**: Cinemagoer 2022.12.27 on Dec 27, 2022

### Import frozen data to PostgreSQL

After getting the cinemagoer repostory, you can import the data by using the below python code file.

[https://raw.githubusercontent.com/cinemagoer/cinemagoer/master/bin/imdbpy2sql.py](https://raw.githubusercontent.com/cinemagoer/cinemagoer/master/bin/imdbpy2sql.py)

```
python imdbpy2sql.py -d ~\Download\imdb-frozendata(JUST GIVE THE FROZENDATA PATH HERE!) -u postgres://user:password@localhost/imdb
```
**Note**: It will take time.

Now, your environment and imdb database is ready to use for query benchmarking.

### Benchmark Queries

For the Join Order Benchmark (JOB) queries, below repostory is used. However, since some queries are not proper for this study' case, some queries are updated. You can also access the final version of queries from below link.

The Join Order Benchmark (JOB) queries from: "How Good Are Query Optimizers, Really?" by Viktor Leis, Andrey Gubichev, Atans Mirchev, Peter Boncz, Alfons Kemper, Thomas Neumann, PVLDB Volume 9, No. 3, 2015, (http://www.vldb.org/pvldb/vol9/p204-leis.pdf)

**Repostory Link**: (https://github.com/gregrahn/join-order-benchmark)

You can access the edited version from this repostory, in ".\queries" folder.

### Import queries to PostgreSQL Database

Our edited version of SQL queries can be imported to Database by below code.

```
python queries\queries2db.py
```

***

### The optimization of join order sequence pattern by using Reinforcement Learning model

In this repository, the study for the ReJoin optimizer has been divided into multiple parts. Up to now, the environment creation process has been successfully completed, and the setup has been done without any issues. For the other part of this study, the ReJoin optimizer is a deep reinforcement learning-based approach that aims to identify the optimal join order sequence for SQL queries. 

However, during the course of the study, some problems were encountered while running the ReJoin optimizer. As a result, the repository was re-implemented by incorporating the exact version of all libraries and an edited version with appropriate directory structures. The aim of this re-implementation was to resolve the issues and ensure that the optimizer functions effectively.

In addition to these efforts, several experiments were conducted by modifying the neural network structure used by the optimizer. These experiments were aimed at investigating the effect of different neural network structures on the performance of the optimizer. The details of these experiments, including the modifications made to the network structure and the results obtained from the experiments, can be found in the repository.

### Running the Train steps and Testing Single Query

- Train target group 4 for 200 episodes
```
python main.py -e 200 -g 1 -tg 4 -se 100 -s ./saved_model/group4-200/
```

Now the plots are in ./outputs folder (default) and the model in  ./saved_model/ 


- Restore saved model and test group 4 
```
python main.py -e 3 -g 1 -tg 4 -r ./saved_model/group4-200/ --testing -o ./outputs/testing/
```

- Restore saved model and keep training on group 5 for 500 episodes
```
python main.py -e 200 -g 1 -tg 5 -se 100 -r ./saved_model/group4-200/ -s ./saved_model/group5-500/
```

- Execute a single query 
```
python main.py --query 3a --episodes 150
```

### Program parameters

<table>
<thead><tr><th>Parameter</th><th>Flag</th><th>Default</th><th>Description</th></tr></thead>
<tbody>
<tr>
<td>Agent configuration file</td><td><code>-a</code>, <code>--agent-config</code></td><td><code>config/ppo.json</code></td><td>Specifies the path to the agent configuration file.</td></tr>
<tr><td>Network specification file</td><td><code>-n</code>, <code>--network-spec</code></td><td><code>config/complex-network.json</code></td><td>Specifies the path to the network specification file.</td></tr>
<tr><td>Number of episodes</td><td><code>-e</code>, <code>--episodes</code></td><td><code>800</code></td><td>Specifies the number of episodes to run during training.</td></tr>
<tr><td>Total groups of different number of relations</td><td><code>-g</code>, <code>--groups</code></td><td><code>1</code></td><td>Specifies the total number of groups with different numbers of relations.</td></tr>
<tr><td>Run specific group</td><td><code>-tg</code>, <code>--target_group</code></td><td><code>5</code></td><td>Specifies the group number to run.</td></tr>
<tr><td>Incremental Mode</td><td><code>-m</code>, <code>--mode</code></td><td><code>round</code></td><td>Specifies the incremental mode to use.</td></tr>
<tr><td>Maximum number of timesteps per episode</td><td><code>-ti</code>, <code>--max-timesteps</code></td><td><code>20</code></td><td>Specifies the maximum number of timesteps per episode.</td></tr>
<tr><td>Run specific query</td><td><code>-q</code>, <code>--query</code></td><td><code>""</code></td><td>Specifies the query to run.</td></tr>
<tr><td>Save agent to this dir</td><td><code>-s</code>, <code>--save_agent</code></td><td></td><td>Specifies the directory to save the agent to.</td></tr>
<tr><td>Restore Agent from this dir</td><td><code>-r</code>, <code>--restore_agent</code></td><td></td><td>Specifies the directory to restore the agent from.</td></tr>
<tr><td>Test agent without learning (use deterministic)</td><td><code>-t</code>, <code>--testing</code></td><td>action=<code>store_true</code> default=<code>False</code></td><td>Specifies whether to test the agent without learning.</td></tr>
<tr><td>Order queries by relations_num</td><td><code>-all</code>, <code>--run_all</code></td><td><code>False</code></td><td>Specifies whether to order queries by the number of relations.</td></tr>
<tr><td>Save agent every x episodes</td><td><code>-se</code>, <code>--save-episodes</code></td><td><code>100</code></td><td>Specifies the frequency at which to save the agent during training.</td></tr>
<tr><td>Select phase (1 or 2)</td><td><code>-p</code>, <code>--phase</code></td><td><code>1</code></td><td>Specifies the training phase to run.</td>
</tr>
</tbody>
</table>

<h3>References</h3>
<ul>

<li><p>Marcus, R., &amp; Papaemmanouil, O. (2018). ReJoin: Deep Reinforcement Learning for Join Order Enumeration. Proceedings of the 2018 International Conference on Advances in Big Data Analytics, 15-19. <a href="https://www.cs.brandeis.edu/~olga/publications/ReJOIN_aiDM18.pdf" target="_new">https://www.cs.brandeis.edu/~olga/publications/ReJOIN_aiDM18.pdf</a></p></li>
<li><p>Join Order Benchmark by Greg Rahn. <a href="https://github.com/gregrahn/join-order-benchmark/tree/master" target="_new">https://github.com/gregrahn/join-order-benchmark/tree/master</a></p></li>
<li><p>Cinemagoer, a Join Order Benchmark. <a href="https://github.com/cinemagoer/cinemagoer" target="_new">https://github.com/cinemagoer/cinemagoer</a></p></li>

</ul>

<h3>Questions</h3>
<p>Please contact Mahsun Altın (<a href="mailto:mhsnltn@gmail.com" target="_new">mhsnltn@gmail.com</a>) if you have any questions.</p></div>
