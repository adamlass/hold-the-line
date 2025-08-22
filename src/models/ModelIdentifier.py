
import os
import wandb
from dotenv import load_dotenv

load_dotenv()

WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")


class ModelIdentifier:
    _run_name: str = None
    _step: int = None
    _location: str = None
    run_id: str = None

    def __init__(self, run_name: str, step: int):
        self._run_name = run_name
        self._step = step

    def __init__(self, model_identifier: str):
        split = model_identifier.split("_")
        assert len(split) == 2, f"Model identifier {model_identifier} is not in the correct format. Should be in the form '[run_name]_[checkpoint]'."
        
        self._run_name, self._step = split
        self._step = int(self._step)
        
        # find the run id of the model
        runs = wandb.Api().runs(f"{WANDB_ENTITY}/{WANDB_PROJECT_NAME}", {"display_name": self.run_name})
        assert len(runs) == 1, f"Found {len(runs)} runs with name {self.run_name}. Expected 1."
        run = runs[0]
        self.run_id = run.id

    def __str__(self):
        return f"{self._run_name}_{self._step}"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        return self._run_name == other.run_name and self._step == other.step
    
    @property
    def step(self) -> int:
        return self._step

    @property
    def run_name(self) -> str:
        return self._run_name