from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngetionConfig:
    root_dir:Path 
    source_url:str
    local_data_file:Path
    unzip_dir:Path
    
    
@dataclass(frozen=True)
class DataValidationConfig:
    root_dir:Path
    STATUS_FILE:str
    ALL_REQUIRED_FILES:list
    
@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir:Path
    STATUS_FILE:str
    ALL_REQUIRED_FILES:list
    
    
@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: str 
    
    
@dataclass
class DataEmbeddingConfig:
    index_name: str
    model_name: str
    
@dataclass
class ModelTrainerConfig:
    index_name: str
    model_name: str


