# Hold The Line

## Prerequisites
Install poetry with
```
curl -sSL https://install.python-poetry.org | python3 -
```
or
```
pip install poetry
```


## Setup guide
1. Setup a file called .env with the following contents:

```
WANDB_PROJECT_NAME=
WANDB_ENTITY=
OPENAI_API_KEY=
HUGGING_FACE_HUB_TOKEN=
```

2. Run `poetry install`


## Run the training 
```
poetry run python src/training/Trainer.py
```

If you are unsure about what arguments to provide, please run:
```
poetry run python src/training/Trainer.py --help
```
