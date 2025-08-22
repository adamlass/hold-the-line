from enum import Enum
from dataclasses import dataclass

from pydantic import BaseModel

class SegmentType(str, Enum):
    info = "info"
    welcome = "welcome"
    dialOptions = "dialOptions"
    holdTheLine = "holdTheLine"
    
    def __repr__(self):
        return f"{self.value}"
    

@dataclass 
class Content():
    text: str
    fileID: str
    
@dataclass
class Segment():
    type: SegmentType
    contents: list[Content]
    nextSegment: "Segment" = None
    id: str = None
    navigationGoals: list[str] = None
    
@dataclass
class DialOption(Content):
    index: int
    nextSegment: Segment
    
class OutputField(BaseModel):
    reference: str
    text: str
    
class Output(BaseModel):
    fields: list[OutputField]

    
class GoalsOutput(BaseModel):
    goals: list[str]