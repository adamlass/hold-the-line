# Hold The Line
<img src="./imgs//Hold The Line.png" alt="Hold The Line Icon"/>

## Description
This repository contain all the files related to my masters thesis titled "Patience Is All You Need: Assessing the Impact of Partial Observations when Fine-Tuning LLMs using Reinforcement Learning". Here, we constructed Hold The Line, a simulation framework that helped us test the impact of partial observations. The project also contains a [demo app](./demo/), "vibe-coded" using Claude Opus 4.1, which is not covered in the thesis since it was only created for visualisation purposes.

> NOTE: Because of publicity and LFS issues, the repo was manually copied from the original at github.itu.dk, which means that the git history is not available here.

## Overview
For fine-tuning logic, please refer to [the Trainer file](src/training/Trainer.py).
The generated data used for fine-tuning is located [here](./data_processed/trees) and is based on the list of companies found in [this list of companies](data_processed/Bay-Area-Companies-List_processed.csv).
For data generation, please reference the Call Tree Generation Pipeline figure from the project to [the list of scripts](./src/scripts/):
<img src="./imgs/Call Tree Generation Pipeline.png" alt="Call Tree Generation Pipeline" width=500/>


### Folder Structure
For extensive details on the projects contents, please refer to the descriptions in our full folder structure:

``` javascript
.
├── data_processed
│   ├── Bay-Area-Companies-List_processed.csv // cleaned list of bay area companies (no duplicates)
│   ├── sources
│   │   └── Bay-Area-Companies-List.csv // original list of bay area companies
│   └── trees // all our 751 Call Trees
│       ├── tree_x_x // an example of a Call Tree
│       │   ├── callTree.json // initial Call Tree structure
│       │   ├── callTreeReplaced.json // Call Tree replaced with content
│       │   ├── callTreeReplacedWithGoals.json // final Call Tree with goals
│       │   ├── description.txt // company description used for generation
│       │   └── replacements.json // original structured output from ChatGPT
|       ├── ...
├── demo // demo app (not covered in project)
│   └── ...
├── poetry.lock
├── pyproject.toml // poetry file
├── requirements.txt
├── setup.job // slurm job used to set up the environment on the HPC
├── src
│   ├── __init__.py
│   ├── classes
│   │   ├── Action.py
│   │   ├── ActionTokenResultTree.py
│   │   ├── ActionTokenTree.py
│   │   ├── ActionType.py // action type
│   │   ├── Call.py
│   │   ├── ConditionalBlocker.py
│   │   ├── Dial.py
│   │   ├── Environment.py // environment base class
│   │   ├── Episode.py 
│   │   ├── EpisodeEnvironment.py
│   │   ├── EpisodeStateMachine.py // state machine used during fine-tuning. Defines the rewards function
│   │   ├── Goal.py
│   │   ├── OnlineCallStateMachine.py
│   │   ├── Process.py // process base class
│   │   ├── Sample.py
│   │   ├── StateMachine.py // state machine base class
│   │   ├── ToyEpisodeStateMachine.py
│   │   ├── TrainingProcess.py
│   │   └── __init__.py
│   ├── dataloading
│   │   ├── EpisodeLoader.py // used for loading fresh episodes
│   │   └── __init__.py
│   ├── models
│   │   ├── BarkTTS.py // (old) used to test synthesising audio for the simulation 
│   │   ├── ModelIdentifier.py
│   │   └── __init__.py
│   ├── scripts
│   │   ├── OpenAIService.py // wrapper service for using the OpenAI api 
│   │   ├── generate_call_trees.py // Call Tree Generator & Dial Option Generator
│   │   ├── generate_folders_with_descriptions.py // scaffolds the initial folder structure based the company list
│   │   ├── generate_goals.py // User Goal Generator
│   │   ├── old // a collection of old scripts used during research 
│   │   │   └── ...
│   │   ├── silero_vad_iterator.py // taken from WhisperStreaming at https://github.com/ufal/whisper_streaming
│   │   └── whisper_online.py // taken from WhisperStreaming at https://github.com/ufal/whisper_streaming
│   ├── services
│   │   ├── CallService.py
│   │   ├── FakeCallServiceStub.py
│   │   ├── LLMService.py // main service for interacting with the LLM during inference and token generation
│   │   ├── TranscriptionService.py
│   │   └── __init__.py
│   ├── training
│   │   ├── Trainer.py // main file for orchestrating the fine-tuning process 
│   │   ├── __init__.py
│   │   └── episode_storage.py // used by the demo app to collect samples
│   ├── transceivers // list of transceiver modules, some of which is used in our inference and training loop
│   │   ├── CallClient.py 
│   │   ├── LLMAgent.py // main logic for the LLMAgent 
│   │   ├── Listener.py
│   │   ├── MotorFunction.py 
│   │   ├── Printer.py
│   │   ├── Transceiver.py // transceiver base class
│   │   ├── Transcriber.py
│   │   └── __init__.py
│   └── utils
│       └── __init__.py // util functions used in the project
├── train.job // slurm job used to start the fine-tuning on the HPC. The arguments used here is representative of a fine-tuning with partial observations
└── tts.job // (old) slurm job used to test synthesising audio for the simulation 
```

## Getting started

### Prerequisites
This project mainly uses poetry for dependency management.

You can install poetry with
```
curl -sSL https://install.python-poetry.org | python3 -
```
or
```
pip install poetry
```


### Setup
1. Please set up a file called .env with the following contents:

```
WANDB_PROJECT_NAME=
WANDB_ENTITY=
OPENAI_API_KEY=
HUGGING_FACE_HUB_TOKEN=
```

The `OPENAI_API_KEY` is only required if you wish to re-generate the data.

2. Run `poetry install` to install all the required dependencies for this project.


### Run the fine-tuning 
```
poetry run python src/training/Trainer.py
```

If you are unsure about what arguments to provide, please run:
```
poetry run python src/training/Trainer.py --help
```
or look at which ones we used for fine-tuning with partial observations in [the training job slurm file](./train.job).


### Run the data generation pipeline
If you wish to re-run the data generation pipeline, please clear the contents in data_processed/trees/* and run the following command:
```
poetry run python src/scripts/generate_folders_with_descriptions.py 
poetry run python src/scripts/generate_call_trees.py 
poetry run python src/scripts/generate_goals.py 
```

## Citations
Our PPO implementation is inspired by GLAM and TWOSOME.
```
@inproceedings{carta2023grounding,
  title={Grounding large language models in interactive environments with online reinforcement learning},
  author={Carta, Thomas and Romac, Cl{\'e}ment and Wolf, Thomas and Lamprier, Sylvain and Sigaud, Olivier and Oudeyer, Pierre-Yves},
  booktitle={International Conference on Machine Learning},
  pages={3676--3713},
  year={2023},
  organization={PMLR}
}
```
```
@article{tan2024true,
  title={True Knowledge Comes from Practice: Aligning Large Language Models with Embodied Environments via Reinforcement Learning},
  author={Weihao Tan and Wentao Zhang and Shanqi Liu and Longtao Zheng and Xinrun Wang and Bo An},
  journal={arXiv preprint arXiv:2401.14151},
  year={2024}
}
```

The ASR code we used was copied from WhisperStreaming:

```
@inproceedings{machacek-etal-2023-turning,
    title = "Turning Whisper into Real-Time Transcription System",
    author = "Mach{\'a}{\v{c}}ek, Dominik  and
      Dabre, Raj  and
      Bojar, Ond{\v{r}}ej",
    editor = "Saha, Sriparna  and
      Sujaini, Herry",
    booktitle = "Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics: System Demonstrations",
    month = nov,
    year = "2023",
    address = "Bali, Indonesia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.ijcnlp-demo.3",
    pages = "17--24",
}
```