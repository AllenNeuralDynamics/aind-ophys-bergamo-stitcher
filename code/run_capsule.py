from pathlib import Path
from bergamo_stitcher import BergamoSettings, BergamoTiffStitcher
import json
from logging_config import setup_logging


if __name__ == "__main__":
    setup_logging("../results/aind_ophys_bergamo_stitcher.log")
    input_dir = Path("../data/")
    data_dir = [d for d in input_dir.rglob("*ophys") if d.is_dir()][0]
    data_description = next(input_dir.rglob("data_description.json"))
    session_fp = next(input_dir.rglob("session.json"))
    output_dir = Path("../results")
    
    with open(data_description) as j:
        unique_id = json.load(j)["name"]
    unique_id = "_".join(str(unique_id).split("_")[-3:])
    bergamo_settings = BergamoSettings(input_dir = data_dir, output_dir=output_dir, unique_id = unique_id, session_fp = session_fp)
    bergamo_stitcher = BergamoTiffStitcher(bergamo_settings)
    bergamo_stitcher.run_converter()