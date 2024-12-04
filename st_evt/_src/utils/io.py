import glob, os
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path


@dataclass
class ExperimentPath:
    exp_dir: Path
    figures_dir: Path
    config_dir: Path
    data_dir: Path
    figures_eda_dir: Path
    figures_model_dir: Path
    figures_results_dir: Path
    
    def __init__(self, root_dir: str):
        
        self.exp_dir = Path(root_dir)
        
        # initialize directories
        figures_dir = self.exp_dir.joinpath("figures")
        figures_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = figures_dir
        
        figures_eda_dir = self.figures_dir.joinpath("eda")
        figures_eda_dir.mkdir(parents=True, exist_ok=True)
        self.figures_eda_dir = figures_eda_dir
        
        figures_model_dir = self.figures_dir.joinpath("model")
        figures_model_dir.mkdir(parents=True, exist_ok=True)
        self.figures_model_dir = figures_model_dir
        
        figures_results_dir = self.figures_dir.joinpath("results")
        figures_results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_results_dir = figures_results_dir
        
        # initialize directories
        config_dir = self.exp_dir.joinpath("config")
        config_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir = config_dir
        # initialize directories
        
        data_dir = self.exp_dir.joinpath("data")
        data_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = data_dir
        
    

@dataclass
class MyPaths:
    data_raw_dir: Path
    data_clean_dir: Path
    data_results_dir: Path
    figures_dir: Path
    
    @classmethod
    def init_from_dot_env(cls):

        load_dotenv()

        data_raw_dir = Path(os.getenv("RAW_DATA_SAVEDIR"))
        data_clean_dir = Path(os.getenv("CLEAN_DATA_SAVEDIR"))
        data_results_dir = Path(os.getenv("RESULTS_DATA_SAVEDIR"))
        figures_dir = Path(os.getenv("FIGURES_DATA_SAVEDIR"))

        return cls(data_raw_dir, data_clean_dir, data_results_dir, figures_dir)



@dataclass
class MySavePaths:
    base_path: Path
    stage: str = "eda"
    method: str = ""
    region: str = "spain"

    @property
    def full_path(self):
        return self.base_path.joinpath(self.stage).joinpath(self.region).joinpath(self.method)
    
    def make_dir(self):
        self.full_path.mkdir(parents=True, exist_ok=True)
        
        
def get_list_filenames(data_path: str="./", ext: str="*"):
    """Loads a list of file names within a directory
    """
    pattern = f"*{ext}"
    return sorted(glob.glob(os.path.join(data_path, "**", pattern), recursive=True))