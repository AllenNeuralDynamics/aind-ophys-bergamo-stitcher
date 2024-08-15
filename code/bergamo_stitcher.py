import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime as dt
from pathlib import Path
from typing import List

import h5py as h5
import numpy as np
from logging_config import setup_logging
from pydantic import BaseModel, Field
from ScanImageTiffReader import ScanImageTiffReader


class BergamoSettings(BaseModel):
    """Settings required to stitch Bergamo images"""

    input_dir: Path = Field(description="directory of tiff files")
    output_dir: Path = Field(description="where to save the hdf5 file")
    unique_id: str = Field(description="name for data (how it relates to the experiment)")


class BaseStitcher:
    def __init__(self, bergamo_settings: BergamoSettings):
        self.input_dir = bergamo_settings.input_dir
        self.output_dir = bergamo_settings.output_dir
        self.unique_id = bergamo_settings.unique_id

    def write_images(
        self,
        image_data_to_cache: List[np.ndarray],
        initial_frame: int,
        cache_filepath: Path,
    ) -> None:
        """
        Cache images to disk

        Parameters
        ----------
        image_data_to_cache : List[np.ndarray]
            A list of image data to cache
        initial_frame : int
            The index of the first frame to concatenate to the array
        cache_filepath : Path
            The filepath to the cache file (should be a temporary file)

        Returns
        -------
        """
        with h5.File(cache_filepath, "a") as f:
            f["data"].resize(initial_frame + len(image_data_to_cache), axis=0)
            f["data"][
                initial_frame : initial_frame + len(image_data_to_cache)
            ] = image_data_to_cache

    def write_final_output(
        self,
        output_filepath: Path,
        **kwargs: dict,
    ):
        """Writes the final output to disk along with the metadata. Clears the temporary
        hdf5 data file

        Parameters
        ----------
        output_filepath : Path
            The filepath to the output file
        **kwargs : dict
            The metadata to write to disk

        Returns
        -------
        None
        """

        # b/c this seems to take a long time
        logging.info("Writing final output file")
        start_time = dt.now()
        with h5.File(output_filepath, "a") as f:
            for key, value in kwargs.items():
                meta_size = 1
                f.create_dataset(
                    key,
                    (meta_size),
                    maxshape=(meta_size,),
                    dtype=h5.special_dtype(vlen=str),
                )
                f[key].resize(meta_size, axis=0)
                f[key][:] = value
        total_time = (dt.now() - start_time).seconds
        logging.info(f"{total_time} seconds to write data")
        logging.info("Finished...")


class BergamoTiffStitcher(BaseStitcher):
    def __init__(self, bergamo_settings: BergamoSettings):
        super().__init__(bergamo_settings)

    def _get_index(self, file_name: str) -> int:
        """Custom sorting key function to extract the index number from the file name
        (assuming the index is a number)

        Parameters
        ----------
        file_name : str
            The name of the file

        Returns
        -------
        int
            The index number
        """
        try:
            # Extract the index number from the file name (assuming the index is a number)
            return int(
                "".join(filter(str.isdigit, file_name.split("_")[-1].split(".")[0]))
            )
        except ValueError:
            # Return a very large number for files without valid index numbers
            return float("inf")

    def _build_tiff_data_structure(self) -> dict:
        """Builds tiff data structures used for the header data later

        Returns
        -------
        dict
            A dictionary containing the tiff data structure
        list
            Index to filepath mapping
        """

        ## Associate an index with each image
        # Find all the unique stages acquired
        logging.info("Building data structure")
        image_list = list(self.input_dir.glob("*.tif"))
        epoch_dict = {}
        epochs = set(
            [
                "_".join(image_path.name.split("_")[:-1])
                for image_path in self.input_dir.glob("*.tif")
                if "stack" not in image_path.name
            ]
        )
        logging.info("Unique epochs: %s", epochs)
        for epoch in epochs:
            epoch_dict[epoch] = [
                str(image)
                for image in image_list
                if epoch == "_".join(image.name.split("_")[:-1])
            ]
            epoch_dict[epoch] = sorted(epoch_dict[epoch], key=self._get_index)

        return epoch_dict

    def write_bergamo(
        self,
        epochs: dict,
        image_width: int = 800,
        image_height: int = 800,
    ) -> Path:
        """
        Reads in a list of tiff files from a specified path (initialized above) and converts them
        to a single h5 file

        Parameters
        ----------
        epochs : dict
            A dictionary containg the sorted epochs
        image_width : int, optional
            The width of the image, by default 800
        image_height : int, optional
            The height of the image, by default 800

        Returns
        -------
        Path
            converted filepath
        tuple
            image shape
        """
        start_time = dt.now()
        epoch_count = 0
        start_epoch_count = 0
        epochs_count = 0
        epochs_location = {}
        header_data = {}
        output_filepath = self.output_dir / f"{self.unique_id}.h5"
        with h5.File(output_filepath, "w") as f:
            f.create_dataset(
                "data",
                (0, image_width, image_height),
                chunks=True,
                maxshape=(None, image_width, image_height),
                dtype="int16",
            )
        # metadata dictionary that keeps track of the epoch name and the location of the
        # epoch image in the stack
        epoch_mapping = self.__get_epoch_mapping()
        epoch_slice_location = {}
        for epoch, _ in epoch_mapping.items():
            for i in epoch_mapping[epoch]:
                header_counter = 0
                for filename in epochs[i]:
                    epoch_name = "_".join(os.path.basename(filename).split("_")[:-1])
                    image_data = ScanImageTiffReader(str(filename)).data()
                    if header_counter == 0:
                        header_data[epoch_name] = ScanImageTiffReader(
                            str(filename)
                        ).metadata()
                    image_shape = image_data.shape
                    frame_count = image_shape[0]
                    self.write_images(image_data, epoch_count, output_filepath)
                    epoch_count += frame_count
                    header_counter += frame_count
                epoch_slice_location[epoch_name] = [start_epoch_count, epoch_count - 1]
                start_epoch_count = epoch_count
            epochs_location[epoch] = [epochs_count, epoch_count - 1]
            epochs_count += epoch_count
        self.write_final_output(
            output_filepath,
            tiff_stem_location=json.dumps(epoch_slice_location),
            epoch_filenames=json.dumps(epochs),
            epoch_location=json.dumps(epochs_location),
            metadata=json.dumps(header_data),
        )

        # Make reference image
        behavior_stem = epoch_mapping["single neuron BCI conditioning"][0]
        len_of_behavior = (
            epoch_slice_location[behavior_stem][1]
            - epoch_slice_location[behavior_stem][0]
            + 1
        )
        vsource = h5.VirtualSource(
            output_filepath, "data", shape=(epoch_count, image_width, image_height)
        )
        layout = h5.VirtualLayout(
            shape=(
                epochs_location["single neuron BCI conditioning"][1]
                - epochs_location["single neuron BCI conditioning"][0]
                + 1,
                image_width,
                image_height,
            ),
            dtype="int16",
        )
        layout[0:len_of_behavior-1, :, :] = vsource[
            epochs_location["single neuron BCI conditioning"][0] : epochs_location[
                "single neuron BCI conditioning"
            ][1],
            :,
            :,
        ]
        with h5.File(
            output_filepath.parent / "reference_image.h5", "w", libver="latest"
        ) as f:
            f.create_virtual_dataset("data", layout, fillvalue=0)

        # Make movie alias
        if not epochs_location.get("2p photostimulation", ""):
            return self.output_dir / f"{self.unique_id}.h5", image_shape
        total_length = 0
        for epoch, epoch_index in epochs_location.items():
            if epoch != "2p photostimulation":
                total_length += epoch_index[1] - epoch_index[0] + 1
        vsource = h5.VirtualSource(
            output_filepath, "data", shape=(epoch_count, image_width, image_height)
        )
        layout = h5.VirtualLayout(
            shape=(total_length, image_width, image_height), dtype="int16"
        )
        offset = 0
        epoch_vset = {}
        new_epoch_location = {}
        for epoch, epoch_index in epochs_location.items():
            if epoch != "2p photostimulation":
                length = epoch_index[1] - epoch_index[0] + 1
                layout[offset : offset + length - 1, :, :] = vsource[
                    epoch_index[0] : epoch_index[1], :, :
                ]
                epoch_vset[epoch] = [offset, offset + length - 1]
                for k, v in epoch_mapping.items():
                    if k == epoch:
                        t = offset
                        for i in v:
                            new_epoch_location[i] = [t, t + length - 1]
                            t += length
                offset += length
            else:
                pass

        with h5.File(output_filepath.parent / "VDS.h5", "w", libver="latest") as f:
            f.create_virtual_dataset("data", layout, fillvalue=0)
            f.create_dataset("epoch_location", data=json.dumps(epoch_vset))
            f.create_dataset("tiff_stem_location", data=json.dumps(new_epoch_location))

        total_time = dt.now() - start_time
        logging.info("Time to cache %s seconds", total_time.total_seconds())

        return self.output_dir / f"{self.unique_id}.h5", image_shape

    def __get_epoch_mapping(self) -> dict:
        """Maps the epoch to the stem of the tiff file

        Returns
        -------
        dict
            A dictionary of the mapped epochs
        """
        epoch_mapping = defaultdict(list)
        session_fp = next(Path("../data").rglob("session.json"), "")
        print(session_fp)
        if not session_fp:
            raise FileNotFoundError("session.json not found")
        with open(session_fp, "rb") as f:
            session_data = json.load(f)
        for epoch in session_data["stimulus_epochs"]:
            epoch_mapping[epoch["stimulus_name"]].append(
                epoch["output_parameters"]["tiff_stem"]
            )
        return epoch_mapping

    def run_converter(self) -> Path:
        """
        Reads in a list of tiff files from a specified path (initialized above) and converts them
        to a single h5 file. Writes relevant metadata to the h5 file.

        Returns
        -------
        Path
            converted filepath
        """

        # Convert the file and build the final metadata structure
        epochs = self._build_tiff_data_structure()

        # metadata dictionary where the keys are the image filename and the
        # values are the index of the order in which the image was read, which
        # epoch it's associated with,  the location of the image in the h5 stack and the
        # image shape
        # tmp_file = TemporaryFile(prefix=self.unique_id, suffix=".h5")
        output_filepath = self.write_bergamo(
            epochs=epochs,
            image_width=800,
            image_height=800,
        )
        return output_filepath

    @classmethod
    def from_args(cls, args: list):
        """
        Adds ability to construct settings from a list of arguments.
        Parameters
        ----------
        args : list
        A list of command line arguments to parse.
        """

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-i",
            "--input-dir",
            required=True,
            type=str,
            help=(
                """
                data-directory for bergamo settings
                """
            ),
        )
        parser.add_argument(
            "-o",
            "--output-dir",
            required=False,
            default=None,
            type=str,
            help=(
                """
                output-directory for bergamo settings
                """
            ),
        )
        parser.add_argument(
            "-u",
            "--unique-id",
            required=False,
            default=None,
            type=str,
            help=(
                """
                unique name for h5 file
                """
            ),
        )
        job_args = parser.parse_args(args)
        job_settings = BergamoSettings(
            input_dir=Path(job_args.input_dir),
            output_dir=Path(job_args.output_dir),
            unique_id=job_args.unique_id,
        )
        return cls(
            job_settings,
        )


if __name__ == "__main__":
    setup_logging("aind_ophys_bergamo_stitcher.log")
    sys_args = sys.argv[1:]
    logging.info("Started job...")
    runner = BergamoTiffStitcher.from_args(sys_args)
    runner.run_converter()
    logging.info("Finished job...")
