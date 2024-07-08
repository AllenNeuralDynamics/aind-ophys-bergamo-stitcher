import argparse
import json
import logging
import os
import sys
from datetime import datetime as dt
from pathlib import Path
from typing import List

import h5py as h5
import numpy as np
from pydantic import BaseModel, Field
from ScanImageTiffReader import ScanImageTiffReader


class BergamoSettings(BaseModel):
    """Settings required to stitch Bergamo images"""

    input_dir: str = Field(description="directory of tiff files")
    output_dir: str = Field(description="where to save the hdf5 file")
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
            input_dir=job_args.input_dir,
            temp_dir=job_args.temp_dir,
            output_dir=job_args.output_dir,
        )
        return cls(
            job_settings=job_settings,
        )


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
        logging.info("Unique epochs: %s".format(epochs))
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
        cache_size=100,
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
        cache_size : int, optional
            The number of images to cache in memory, by default 100
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
        # to keep track of all the images stored to disk
        total_count = 0
        # to keep track of the number of images stored in memory
        images_stored = 0
        # to keep track of the number of images to store to disk when there is buffer overflow
        frames_to_store = 0
        image_buffer = np.zeros((cache_size, image_width, image_height))
        output_filepath = self.output_dir / f"{self.unique_id}.h5"
        start_epoch_count = 0
        with h5.File(output_filepath, "w") as f:
            f.create_dataset(
                "data",
                (0, 800, 800),
                chunks=True,
                maxshape=(None, 800, 800),
            )
        # metadata dictionary that keeps track of the epoch name and the location of the
        # epoch image in the stack
        epoch_slice_location = {}
        lookup_table = {}
        for epoch in epochs.keys():
            for filename in epochs[epoch]:
                epoch_name = "_".join(os.path.basename(filename).split("_")[:-1])
                image_shape = ScanImageTiffReader(str(filename)).shape()
                image_data = ScanImageTiffReader(str(filename)).data()
                lookup_table[filename] = {}
                lookup_table[filename]["image_shape"] = image_shape
                lookup_table[filename]["epoch"] = epoch_name
                # Grabbing the epoch name to keep track of changes and
                # index position of each epoch in the stack. Will compare to previous_epoch_name
                if image_shape[0] + images_stored >= cache_size:
                    frames_to_store = cache_size - images_stored
                    image_buffer[images_stored:cache_size] = image_data[:frames_to_store]
                    total_count += frames_to_store
                    self.write_images(
                        image_buffer, total_count - cache_size, output_filepath
                    )
                    images_stored = 0
                    image_buffer = np.zeros((cache_size, image_width, image_height))
                    image_buffer[images_stored : image_shape[0] - frames_to_store] = (
                        image_data[frames_to_store:]
                    )
                    images_stored += image_shape[0] - frames_to_store
                    total_count += image_shape[0] - frames_to_store
                else:
                    image_buffer[images_stored : images_stored + image_shape[0]] = (
                        image_data
                    )
                    images_stored += image_shape[0]
                    total_count += image_shape[0]

                lookup_table[filename]["location_in_stack"] = [
                    total_count,
                    total_count + frames_to_store,
                ]

            # save the last epoch slice location
            epoch_slice_location[epoch_name] = []
            epoch_slice_location[epoch_name].append((start_epoch_count, total_count))
            start_epoch_count = total_count + 1
        # if images did not get cached to disk, cache them now
        if images_stored > 0:
            final_buffer = np.zeros((images_stored, image_width, image_height))
            final_buffer[:images_stored] = image_buffer[:images_stored]
            self.write_images(final_buffer, total_count - images_stored, output_filepath)
        print(f"Total count {total_count}")
        self.write_final_output(
            output_filepath,
            epoch_slice_location=json.dumps(epoch_slice_location),
            epoch_filenames=json.dumps(epochs),
            lookup_table=json.dumps(lookup_table),
        )
        total_time = dt.now() - start_time
        print(f"Time to cache {total_time.seconds} seconds")
        return self.output_dir / f"{self.unique_id}.h5", image_shape

    def run_converter(self, chunk_size=500) -> Path:
        """
        Reads in a list of tiff files from a specified path (initialized above) and converts them
        to a single h5 file. Writes relevant metadata to the h5 file.

        Parameters
        ----------
        chunk_size : int, optional
            The chunk size to write to disk, by default 500
        Returns
        -------
        Path
            converted filepath
        """

        # Convert the file and build the final metadata structure
        epochs, image_shape = self._build_tiff_data_structure()

        # metadata dictionary where the keys are the image filename and the
        # values are the index of the order in which the image was read, which
        # epoch it's associated with,  the location of the image in the h5 stack and the
        # image shape
        # tmp_file = TemporaryFile(prefix=self.unique_id, suffix=".h5")
        output_filepath = self.write_bergamo(
            cache_size=chunk_size,
            epochs=epochs,
            image_width=image_shape[0],
            image_height=image_shape[1],
        )
        # write stack to h5
        # stack_fp = next(self.input_dir.glob("stack*.tif"), None)
        # if stack_fp:
        #     with ScanImageTiffReader(str(stack_fp)) as reader:
        #         stack_data = reader.data()
        #         stack_meta = reader.metadata()
        #     with h5.File(self.output_dir / "stack.h5", "w") as f:
        #         f.create_dataset(
        #             "data",
        #             (0, 800, 800),
        #             maxshape=(None, 800, 800),
        #         )
        #     self.write_images(stack_data, 0, self.output_dir / "stack.h5")
        #     self.write_final_output(
        #         self.output_dir / "stack.h5",
        #         metadata=json.dumps(stack_meta),
        #     )

        return output_filepath


if __name__ == "__main__":
    sys_args = sys.argv[1:]
    logging.info("Started job...")
    runner = BergamoSettings.from_args(sys_args)
    runner.run_converter()
    logging.info("Finished job...")