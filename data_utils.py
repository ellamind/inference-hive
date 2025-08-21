from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import time

import datasets as hfds
from loguru import logger
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as pads


@dataclass
class CheckpointFile:
    file_path: Path
    num_rows: int
    file_size: int


class DatasetWriter:
    def __init__(
        self,
        dataset_dir: Path,
        schema: pa.Schema,
        shard: int,
        max_file_size_mb: int = 2000,
        max_rows_per_file: int = None,
        batch_size: int = 1_000,
        checkpoint_interval_seconds: int = 1800,
        check_time_every_n_rows: int = 100,
    ) -> None:
        self.schema = schema
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024  # Convert MB to bytes
        self.max_rows_per_file = max_rows_per_file
        self.batch_size = batch_size
        self.checkpoint_interval_seconds = checkpoint_interval_seconds
        self.check_time_every_n_rows = check_time_every_n_rows
        self.shard_name = str(f"shard{shard:06d}")
        self.dataset_dir = dataset_dir.absolute()
        self.dataset_dir.mkdir(exist_ok=True)

        self._last_checkpoint_time = None

        self.__init_tempfile()
        self.__init_batch()

        self.rows_added = 0
        self._closed = False
        
        self.checkpoint_files = self._get_existing_checkpoints()
        
    def _should_checkpoint_by_time(self) -> bool:
        """Check if checkpoint interval has been reached using simple time comparison"""
        if self.checkpoint_interval_seconds <= 0:
            return False
            
        current_time = time.time()
        if self._last_checkpoint_time is None:
            self._last_checkpoint_time = current_time
            return False
            
        time_since_last_checkpoint = current_time - self._last_checkpoint_time
        return time_since_last_checkpoint >= self.checkpoint_interval_seconds

    def __init_tempfile(self):
        self.tempfile = self.dataset_dir / f"{self.shard_name}.incomplete"
        self.pq_writer = pq.ParquetWriter(self.tempfile, self.schema, compression="zstd")
        self.rows_added_to_tempfile = 0
        self._last_checkpoint_time = time.time()

    def __init_batch(self):
        self.batch = {col: [] for col in self.schema.names}
        self.rows_in_batch = 0

    def _get_tempfile_size(self):
        if self.tempfile.exists():
            return self.tempfile.stat().st_size
        return 0

    def add_row(self, row: Dict[str, Any]):
        for col in self.schema.names:
            assert col in row, f"row to add is missing key from schema: {col}. Keys in row: {row.keys()}"
            self.batch[col].append(row[col])
        self.rows_in_batch += 1
        self.rows_added += 1

        should_write_batch = self.rows_in_batch >= self.batch_size
        
        if should_write_batch or (self.rows_added % self.check_time_every_n_rows == 0 and self._should_checkpoint_by_time() and self.rows_in_batch > 0):
            self.write_batch()

    def write_batch(self, emergency=False, final=False):
        self.pq_writer.write_batch(batch=pa.RecordBatch.from_pydict(self.batch, schema=self.schema))
        self.rows_added_to_tempfile += self.rows_in_batch
        self.__init_batch()
        
        if emergency:
            logger.info("Creating checkpoint because of emergency.")
            self.create_checkpoint()
        elif final:
            logger.info("Finalizing.")
            logger.info("Creating checkpoint to finalize.")
            self.create_checkpoint(final=True)
        else:
            # Check if we should create a checkpoint
            size = self._get_tempfile_size() + sum(checkpoint_file.file_size for checkpoint_file in self.checkpoint_files)
            size_limit_exceeded = size >= self.max_file_size_bytes

            rows = self.rows_added_to_tempfile + sum(checkpoint_file.num_rows for checkpoint_file in self.checkpoint_files)
            row_limit_exceeded = (self.max_rows_per_file is not None and self.max_rows_per_file > 0 and rows >= self.max_rows_per_file)
        
            checkpoint_interval_exceeded = self._should_checkpoint_by_time()

            if checkpoint_interval_exceeded or size_limit_exceeded or row_limit_exceeded:
                logger.info(f"Creating checkpoint because checkpoint_interval={checkpoint_interval_exceeded} or size_limit={size_limit_exceeded} or row_limit={row_limit_exceeded}.")
                self.create_checkpoint()
            

    def create_checkpoint(self, emergency=False, final=False):
        if self.pq_writer is not None:
            self.pq_writer.close()
            self.pq_writer = None
        
        # Move current temp file to checkpoint file
        checkpoint_file = self.get_checkpoint_file()
        self.tempfile.rename(checkpoint_file)
        if emergency:
            return
        self.checkpoint_files.append(CheckpointFile(checkpoint_file, self.rows_added_to_tempfile, checkpoint_file.stat().st_size))
        
        # check if we should create a part
        size = sum(checkpoint_file.file_size for checkpoint_file in self.checkpoint_files)
        size_limit_exceeded = size >= self.max_file_size_bytes

        rows = sum(checkpoint_file.num_rows for checkpoint_file in self.checkpoint_files)
        row_limit_exceeded = (self.max_rows_per_file is not None and self.max_rows_per_file > 0 and rows >= self.max_rows_per_file)
        
        if size_limit_exceeded or row_limit_exceeded or final:
            logger.info(f"Creating part because size_limit={size_limit_exceeded} or row_limit={row_limit_exceeded} or final={final}.")
            self.create_part()
        if not final:
            self.__init_tempfile()
    
    def create_part(self):
        
        files_to_merge = [checkpoint_file.file_path for checkpoint_file in self.checkpoint_files]
        logger.info(f"Creating part from {len(self.checkpoint_files)} checkpoint files")
        
        part_file = self.get_part_file()
        
        try:
            if len(files_to_merge) == 1:
                files_to_merge[0].rename(part_file)
            else:
                self._merge_parquet_files(files_to_merge, part_file)
            
            self.checkpoint_files = []
                
        except Exception as e:
            logger.error(f"Failed to create part file {part_file} from checkpoint files {files_to_merge}\n\n{e}")
            raise
    
    def _merge_parquet_files(self, input_files, output_file):
        # Use .incomplete extension for atomic operation
        temp_output_file = output_file.with_suffix('.incomplete')
        
        try:
            dataset = pads.dataset(input_files)
            with pq.ParquetWriter(temp_output_file, self.schema, compression="zstd") as writer:
                for batch in dataset.to_batches():
                    writer.write_batch(batch)
            
            # Atomic rename - only happens if merge was successful
            temp_output_file.rename(output_file)
            
            # Only delete input files after successful merge and rename
            for file in input_files:
                file.unlink()
                
        except Exception as e:
            # Clean up incomplete file on failure
            if temp_output_file.exists():
                temp_output_file.unlink()
            logger.error(f"Failed to merge parquet files: {e}")
            raise

    def get_part_file(self):
        next_part_id = 0
        while True:
            f = self.dataset_dir / f"{self.shard_name}_part{next_part_id:06d}.parquet"
            if f.exists():
                next_part_id += 1
            else:
                return f

    def get_checkpoint_file(self):
        next_checkpoint_id = 0
        while True:
            f = self.dataset_dir / f"{self.shard_name}_checkpoint{next_checkpoint_id:06d}.parquet"
            if f.exists():
                next_checkpoint_id += 1
            else:
                return f
            
    def _get_existing_checkpoints(self):
        checkpoint_pattern = f"{self.shard_name}_checkpoint*.parquet"
        
        existing_checkpoint_files = sorted(list(self.dataset_dir.glob(checkpoint_pattern)))
        checkpoint_files = []
        if existing_checkpoint_files:
            logger.info(f"Found {len(existing_checkpoint_files)} checkpoint files for {self.shard_name}")
            for checkpoint_file in sorted(existing_checkpoint_files):
                file_size = checkpoint_file.stat().st_size
                with pq.ParquetFile(checkpoint_file) as parquet_file:
                    num_rows = parquet_file.metadata.num_rows
                checkpoint_files.append(CheckpointFile(checkpoint_file, num_rows, file_size))
                logger.info(f"Checkpoint: {checkpoint_file.name} ({num_rows:_} rows) ({file_size / 1024 / 1024:.1f}MB)")
        return checkpoint_files

    def close(self, emergency=False):
        if self._closed:
            return
            
        if self.rows_in_batch > 0:
            # write the current data if there is any
            self.write_batch(emergency=emergency, final=not emergency)
        self._closed = True
        logger.info("Writer closed.")
    


class NoDatasetFilesError(Exception):
    pass

class DatasetReader:
    def __init__(self, dataset_dir: Path, shard: int = None, num_shards: int = None, glob_pattern: str = "**/*.parquet", **to_batches_kwargs) -> None:
        if shard is not None and not num_shards:
            raise ValueError("If you specify the shard to load you must also specify total num_shards.")
        
        self.dataset_dir = dataset_dir
        self.dataset_files = sorted(list(self.dataset_dir.glob(glob_pattern)))
        logger.info(f"Found {len(self.dataset_files)} dataset files in {self.dataset_dir}")
        if len(self.dataset_files) == 0:
            raise NoDatasetFilesError(f"No files found in {self.dataset_dir} that match pattern: '**/*.parquet'")

        if num_shards is not None:
            assert num_shards <= len(
                self.dataset_files
            ), f"Cannot have more shards ({num_shards}) than data files ({len(self.dataset_files)})."
            assert (
                shard < num_shards
            ), f"shard out of bounds. {shard=}, {num_shards=}. shard must be in range 0 to {num_shards-1}"
        self.shard = shard
        self.num_shards = num_shards

        if (self.shard is not None) and (self.num_shards is not None):
            self.dataset_files = self.dataset_files[self.shard::self.num_shards]

        self.ds = pads.dataset(self.dataset_files)

        default_to_batches_kwargs = {
            "batch_size": 10_000,
            "batch_readahead": 2,
            "fragment_readahead": 0,
        }
        self.to_batches_kwargs = {**default_to_batches_kwargs, **to_batches_kwargs}

    def __str__(self):
        return f"DatasetReader(dataset_dir={self.dataset_dir}, shard={self.shard}, num_shards={self.num_shards}, num_files={self.num_files()}, num_rows={self.num_rows()}, to_batches_kwargs={self.to_batches_kwargs}, schema={self.get_schema()})"

    def __len__(self) -> int:
        return self.num_rows()
    
    def __iter__(self):
        batches_iter = self.ds.to_batches(**self.to_batches_kwargs)
        for batch in batches_iter:
            batch = batch.to_pylist()
            for row in batch:
                yield row

    def num_rows(self) -> int:
        return self.ds.count_rows()

    def num_files(self) -> int:
        return len(self.dataset_files)

    def get_dataset(self) -> pads.Dataset:
        return self.ds

    def get_schema(self) -> pa.Schema:
        return self.ds.schema

class DatasetWrapper:
    
    def __init__(self, dataset, udf_func, udf_kwargs=None):
        self._wrapped_dataset = dataset
        self._udf_func = udf_func
        self._udf_kwargs = udf_kwargs or {}
    
    def __getattr__(self, name):
        return getattr(self._wrapped_dataset, name)
    
    def __iter__(self):
        """
        Override __iter__ to apply the UDF function to each row.
        """
        for row in self._wrapped_dataset:
            try:
                transformed_row = self._udf_func(row, **self._udf_kwargs)
                yield transformed_row
            except Exception as e:
                logger.error(f"Error applying UDF to row: {e}")
                raise
    
    def __len__(self):
        return len(self._wrapped_dataset)
    
    def __str__(self):
        return f"DatasetWrapper({self._wrapped_dataset})"
    

def wrap_with_udf(ds, config):
    try:
        import udf
        udf_func = getattr(udf, config.apply_udf)
    except ImportError:
        logger.error("Could not import udf module. Make sure udf.py exists in the project root.")
        raise
    except AttributeError:
        logger.error(f"UDF function '{config.apply_udf}' not found in udf.py")
        raise
    
    udf_kwargs = config.apply_udf_kwargs or {}
    
    logger.info(f"Wrapping dataset with UDF: {config.apply_udf}, kwargs: {udf_kwargs}")
    
    return DatasetWrapper(ds, udf_func, udf_kwargs)

def load_data(config, shard: int = None, num_shards: int = None):
    logger.info(
        f"Loading dataset with with kwargs: {config.dataset_kwargs}"
    )
    kwargs = config.dataset_kwargs or {}
    if "hf" in config.dataset_type:
        if config.dataset_type == "hf-disk":
            logger.info("Loading dataset with load_from_disk")
            ds = hfds.load_from_disk(config.dataset_path, **kwargs)
        else:
            logger.info("Loading dataset with load_dataset")
            ds = hfds.load_dataset(str(config.dataset_path), **kwargs)

        # Check if dataset is a DatasetDict and raise error if it is
        if isinstance(ds, (hfds.DatasetDict, hfds.IterableDatasetDict)):
            available_splits = list(ds.keys())
            raise ValueError(
                f"Dataset is a DatasetDict with splits: {available_splits}. "
                "Please specify a split in your dataset_kwargs (e.g., 'split': 'train') "
            )
        
        # Check if dataset is an IterableDataset and raise error, as IterableDataset does not support sharding
        if isinstance(ds, hfds.IterableDataset):
            raise ValueError(
                "IterableDataset is currently not supported. Use a regular Dataset instead (streaming=False)."
            )
        if shard is not None and num_shards is not None:
            ds = ds.shard(num_shards=num_shards, index=shard)
    
    elif config.dataset_type == "parquet":
        logger.info("Loading parquet dataset with DatasetReader")
        ds = DatasetReader(config.dataset_path, shard=shard, num_shards=num_shards, **kwargs)
    else:
        raise ValueError(f"Invalid dataset type: {config.dataset_type}. Supported types are: 'hf', 'hf-disk', 'parquet'")
    if config.apply_udf:
        ds = wrap_with_udf(ds, config)
    return ds