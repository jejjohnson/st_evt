import glob, os
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path


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