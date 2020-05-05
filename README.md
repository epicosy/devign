# Devign

Implementation of Devign Model in Python with code for processing the dataset and generation of Code Property Graphs.
###### This project is under development. For now, just the Abstract Syntax Tree is considered for the graph embedding of code and model training.

## Table of Contents

* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
     * [Software](#software)
     * [Python Libraries](#python-libraries)
  * [Setup](#setup)
* [Structure](#structure)
* [Usage](#usage)
    * [Dataset](#dataset)
	* [Fields](#fields)
    * [Baseline "main.py"](#baseline-mainpy)
        * [Create Task](#create-task)
        * [Embed Task](#embed-task)
        * [Process Task](#process-task)
* [Roadmap](#Roadmap)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Install the necessary dependencies before running the project:
<br/>
##### Software:
* [Joern](https://joern.io/docs/)
* [Python (=>3.6)](https://www.python.org/)
##### Python Libraries:
* [Pandas (>=1.0.1)](https://pandas.pydata.org/)
* [scikit-learn (>=0.22.2)](https://scikit-learn.org/stable/)
* [PyTorch (>=1.4.0)](https://pytorch.org/)
* [PyTorch Geometric (>=1.4.2)](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
* [Gensim (>=3.8.1)](https://radimrehurek.com/gensim/)
* [cpgclientlib (>=0.11.111)](https://pypi.org/project/cpgclientlib/)

### Notes

---
These notes might save you some time:

* Changes to the ```configs.json``` structure need to be reflected in the ```configs.py``` script.
* **PyTorch Geometric** has several dependencies that need to match, including **PyTorch**. 
Follow the installation steps on their website.
* Joern processing might be slow and even freeze your OS, that depends on your system's specifications.
Choose a smaller size for the chunks that are processed when splitting the dataset during the **Create** task.
That can be done by changing the ```"slice_size"``` value under ```"create"``` in the configurations file ```configs.json``` 
* In the ```"slice_size"``` file, the nodes are filtered and discarded if the size is greater than the limit configured.
* When changing the number of nodes considered for processing, ```"nodes_dim"``` under ```"embed"``` 
needs to match ```"in_channels"```, under ```"devign" -> "model" -> "conv_args" -> "conv1d_1"```.
* The embedding size is equal to Word2Vec vector size plus 1.
* When executing the **Create** task, a directory named ```joern``` is created and deleted automatically under ```'project'\data\```.
### Setup

---
###### __For now this project is not pip installable. With the proper use cases will be implemented.__

This section gives the steps, explanations and examples for getting the project running.
#### 1) Clone this repo

``` console
$ git clone https://github.com/epicosy/devign/devign.git
```

#### 2) Install Prerequisites

#### 3) Configure the project
Verify you have the correct directory structure by matching with the ```"paths"``` in the configurations file ```configs.json```.
The dataset related files that are generated are saved under those paths. 
#### 4) Joern
This step is only necessary for the **Create** task.
Follow the instructions on [Joern's documentation page](https://joern.io/docs/) and install Joern's command line tools
under ```'project'\joern\joern-cli\ ```.
<br/>

## Structure

---
``` 
├── LICENSE
├── README.md                       <- The top-level README for developers using this project.
├── data
│   ├── cpg                         <- Dataset with CPGs.
│   ├── input                       <- Cannonical dataset for modeling.
│   ├── model                       <- Trained models.
│   ├── raw                         <- The original, immutable data dump.
│   ├── tokens                      <- Tokens dataset files generated from the raw data functions.
│   └── w2v                         <- Word2Vec model files for initial embeddings.
│
├── joern
│   ├── joern-cli                   <- Joern command line tools for creating and analyzing code property graphs.
│   └── graphs-for-funcs.sc         <- Script that returns in Json format the AST, CGF, and PDG for each method 
│                                       contained in the loaded CPG.
│
├── src                             <- Source code for use in this project.
│   ├── __init__.py                 <- Makes src a Python package.
│   │
│   ├── data                        <- Data handling scripts.
│   │   ├── __init__.py             <- Makes data a Python package.
│   │   └── datamanger.py           <- Module for the most essential operations on the dataset.
│   │
│   ├── prepare                     <- Package for CPG generation and representation.
│   │   ├── __init__.py             <- Makes prepare a Python package.
│   │   ├── cpg_client_wrapper.py   <- Simple class wrapper for the CpgClient that interacts with the Joern REST server 
│   │   ├── cpg_generator.py        <- Ad-hoc script for creating CPGs with Joern and processing the results.
│   │   └── embeddings.py           <- Module that embeds the graph nodes into node features.
│   │
│   ├── process                     <- Scripts for modeling and predictions.
│   │   ├── __init__.py             <- Makes process a Python package.
│   │   ├── devign.py               <- Module that implements the devign model.
│   │   ├── loader_step.py          <- Module for one epoch iteration over dataset
│   │   ├── model.py                <- Module that implements the devign neural network.
│   │   ├── modeling.py             <- Module for training and prediction the model.
│   │   ├── step.py                 <- Module that performs a forward step on a batch for train/val/test loop.
│   │   └── stopping.py             <- Module that performs early stopping.
│   │
│   │
│   └── utils                       <- Package with helper components like functions and classes, used across 
│       │                              the project.
│       ├── __init__.py             <- Makes utils a Python package.
│       ├── log.py                  <- Module for logging modules messages.
│       ├── functions               <- Auxiliar functions for processing.
│       │   ├── __init__.py         <- Makes functions a Python package
│       │   ├── cpg.py              <- Module with auxiliar functions for CPGs.
│       │   ├── digraph.py          <- Module for creating digraphs from nodes.
│       │   └── parase.py           <- Module for parsing source code into tokens.
│       │ 
│       └── objects                 <- Auxiliar data classes with basic methods.
│           ├── __init__.py         <- Makes objects a Python package.
│           ├── cpg                 <- Auxiliar data classes for representing and handling the Json graphs.
│           │   ├── __init__.py     
│           │   ├── ast.py
│           │   ├── edge.py
│           │   ├── function.py
│           │   ├── node.py
│           │   └── properties.py
│           │
│           ├── input_dataset.py    <- Custom wrapper for Torch Dataset.
│           ├── metrics.py          <- Module for evaluating the results.
│           └── stats.py            <- Module for handling raw results.
│       
│
├── configs.py                      <- Configuration management script.
├── configs.json                    <- Project configurations used by main.py. 
└── main.py                         <- Main script file that joins the modules into executable tasks. 
```

##Usage

### Dataset

---
The dataset used is the [partial dataset](https://sites.google.com/view/devign) released by the authors.
The dataset is handled with Pandas and the file ```src/data/datamanger.py``` contains wrapper functions for the most essential operations. 
<br/>
<br/>
A small sample from the original dataset is available for testing purposes. 
The sample dataset contains functions from the **FFmpeg** project with maximum of 287 nodes per function.
For each task, the necessary dataset files are available under the respective folders.
<br/>
<br/>
For example, under ```data/cpg``` are available the datasets with the graphs constituting the CPG for the functions.

#### Fields

|project|               commit_id                  | target |                           func                    |
|------:|:----------------------------------------:| ------:| -------------------------------------------------:|
|FFmpeg | 973b1a6b9070e2bf17d17568cbaf4043ce931f51 |    0   | static av_cold int vdadec_init(AVCodecContext ... |
|FFmpeg | 321b2a9ded0468670b7678b7c098886930ae16b2 |    0   | static int transcode(AVFormatContext **output_... |
|FFmpeg | 5d5de3eba4c7890c2e8077f5b4ae569671d11cf8 |    0   | static void v4l2_free_buffer(void *opaque, uin... |
|FFmpeg | 32bf6550cb9cc9f487a6722fe2bfc272a93c1065 |    0   | int ff_get_wav_header(AVFormatContext *s, AVIO... |
|FFmpeg | 57d77b3963ce1023eaf5ada8cba58b9379405cc8 |    0   | int av_opencl_buffer_write(cl_mem dst_cl_buf, ... |
|       |   ...                                    |   ...  |  ...                                          ... |
|qemu   | 1ea879e5580f63414693655fcf0328559cdce138 |    0   | static int no_init_in (HWVoiceIn *hw, audsetti... |
|qemu   | f74990a5d019751c545e9800a3376b6336e77d38 |    0   | uint32_t HELPER(stfle)(CPUS390XState *env, uin... |
|qemu   | a89f364ae8740dfc31b321eed9ee454e996dc3c1 |    0   | static void pxa2xx_fir_write(void *opaque, hwa... |
|qemu   | 39fb730aed8c5f7b0058845cb9feac0d4b177985 |    0   | static void disas_thumb_insn(CPUARMState *env,... |
|FFmpeg | 7104c23bd1a1dcb8a7d9e2c8838c7ce55c30a331 |    0   | static void rv34_pred_mv(RV34DecContext *r, in... |


### Baseline "main.py"

---
The script ```main.py``` contains functions that put together the modules into executable tasks for the baseline approach.
It can be used as example to elaborate custom functionalities.
<br/>
<br/>
The basic baseline transforms the dataset to the input for the model, proceeding with it's training and evaluation. 
The tasks that compose it are Create, Embed and Process.
``` console
$ python main.py -c -e -p.
```

For each task, verify that the correct files are in the respective folders.
For example, executing the **Process** task requires the input datasets that contain 
the embedded graphs with the associated labels.

#### Create Task
This is the first task where the dataset is filtered (optionally) and augmented with a column that 
contains the respective Code Property Graph (CPG).
<br/>
<br/>
Functions in the dataset are written to files into a target directory which Joern is queried with for creating the CPG. 
After the CPG creation, Joern is queried with the script "graph-for-funcs.sc" which creates the graphs from the CPG.
Those are returned in JSON format, containing all the functions with the respective AST, CFG and PDG graphs.

Execute with:

``` console
$ python main.py -c
```
Filtering the dataset can be done with ```data.apply_filter(raw: pandas.Dataframe, select: callable)``` 
under ```create_task``` function.

#### Embed Task
This task transforms the source code functions into tokens which are used to generate and train the word2vec model 
for the initial embeddings. The nodes embeddings are done as explained in the paper, for now just for the AST: 
![Node Representation](https://lh6.googleusercontent.com/DcfjhgnCH23Zsw7FZ5_WFr2M-4Tzn9uO8U32QpbaUBiBOjfi3-yRE0mDT7SEIEe4OorV8-BvDppk3CGxY8AfPxCAdeZEkkf47K9X_W-mfc5QSCB8VIU=w1175)

Execute with:
``` console
$ python main.py -e
```

#### Process Task
In this task the previous transformed dataset is split into train, validation and test sets which are
 used to train an evaluate the model.

Execute with:
``` console
$ python main.py -p
```

Enable EarlyStopping for training with:

``` console
$ python main.py -pS
``` 

## Roadmap

See the [open issues](https://github.com/epicosy/devign/issues) for a list of proposed features (and known issues).

## Authors
* **Yaqin Zhou**, **Shangqing Liu**, **Jingkai Siow**, **Xiaoning Du**, **Yang Liu**
* *Initial work* - [Devign Paper](https://arxiv.org/pdf/1909.03496v1.pdf), [Node Representation and Datasets](https://sites.google.com/view/devign/home)

## License
Distributed under the MIT License. See LICENSE for more information.

## Contact

Eduard Pinconschi - eduard.pinconschi@tecnico.ulisboa.pt

## Acknowledgments
Guidance and ideas for some parts from:

* [Transformer for Software Vulnerability Detection](https://github.com/hazimhanif/svd-transformer)
* [SySeVR: A Framework for Using Deep Learning to Detect Vulnerabilities](https://github.com/SySeVR/SySeVR)
* [VulDeePecker algorithm implemented in Python](https://github.com/johnb110/VDPython)
