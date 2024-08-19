from pathlib import Path
from bergamo_stitcher import BergamoSettings, BergamoTiffStitcher
import json
from logging_config import setup_logging


if __name__ == "__main__":
    setup_logging("../results/aind_ophys_bergamo_stitcher.log")
    input_dir = Path("../data/")
    try:
        data_dir = next(input_dir.glob("*/ophys"))
    except:
        data_dir = next(input_dir.glob("*ophys"))
    output_dir = Path("../results")
    data_description = next(input_dir.rglob("data_description.json")
    with open(data_description) as j:
        unique_id = json.load(j)["name"]
    unique_id = "_".join(str(unique_id).split("_")[-3:])
    bergamo_settings = BergamoSettings(input_dir = data_dir, output_dir=output_dir, unique_id = unique_id)
    bergamo_stitcher = BergamoTiffStitcher(bergamo_settings)
    bergamo_stitcher.run_converter()