from pathlib import Path
from typing import Any, Dict

import datasets as hfds
from loguru import logger
import numpy as np
# import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as pads


class DatasetWriter:
    def __init__(
        self,
        dataset_dir: Path,
        schema: pa.Schema,
        shard: int,
        max_file_size_mb: int = 2000,
        max_rows_per_file: int = None,
        batch_size: int = 1_000,
    ) -> None:
        self.schema = schema
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024  # Convert MB to bytes
        self.max_rows_per_file = max_rows_per_file
        self.batch_size = batch_size

        self.shard_name = str(f"shard{shard:06d}")
        self.dataset_dir = dataset_dir.absolute()
        self.dataset_dir.mkdir(exist_ok=True)

        self.__init_tempfile()
        self.__init_batch()

        self.num_files_in_shard = 0
        self.rows_added = 0
        self._closed = False

    def __init_tempfile(self):
        # Close existing writer if it exists
        if hasattr(self, 'pq_writer') and self.pq_writer is not None:
            self.pq_writer.close()
        
        self.tempfile = self.dataset_dir / f"{self.shard_name}.incomplete"
        self.pq_writer = pq.ParquetWriter(self.tempfile, self.schema, compression="zstd")
        self.rows_added_to_tempfile = 0

    def __init_batch(self):
        self.batch = {col: [] for col in self.schema.names}
        self.rows_in_batch = 0

    def add_row(self, row: Dict[str, Any]):
        for col in self.schema.names:
            assert col in row, f"row to add is missing key from schema: {col}. Keys in row: {row.keys()}"
            self.batch[col].append(row[col])
        self.rows_in_batch += 1
        self.rows_added += 1

        if self.rows_in_batch >= self.batch_size:
            self.write_batch()

    def write_batch(self):
        self.pq_writer.write_batch(batch=pa.RecordBatch.from_pydict(self.batch, schema=self.schema))
        self.rows_added_to_tempfile += self.rows_in_batch
        self.__init_batch()
        
        size_limit_exceeded = self.tempfile.exists() and self.tempfile.stat().st_size >= self.max_file_size_bytes
        row_limit_exceeded = (self.max_rows_per_file is not None and self.max_rows_per_file > 0 and self.rows_added_to_tempfile >= self.max_rows_per_file)
        
        if size_limit_exceeded or row_limit_exceeded:
            self.commit()

    def get_file_name(self):
        # Keep incrementing until we find a filename that doesn't exist
        while True:
            fname = self.dataset_dir / f"{self.shard_name}_part{self.num_files_in_shard:06d}.zstd.parquet"
            if not fname.exists():
                return fname
            self.num_files_in_shard += 1

    def commit(self, final=False):
        # Make sure the writer is closed before renaming the file
        if self.pq_writer is not None:
            self.pq_writer.close()
            self.pq_writer = None
        
        fname = self.get_file_name()
        self.tempfile.rename(fname)
        self.num_files_in_shard += 1
        if not final:
            self.__init_tempfile()

    def close(self):
        if self._closed:
            # Already closed, nothing to do
            return
            
        if self.rows_in_batch > 0:
            self.write_batch()
        
        if self.pq_writer is not None:
            self.pq_writer.close()
            self.pq_writer = None
        
        if hasattr(self, 'tempfile') and self.tempfile.exists():
            if self.rows_added_to_tempfile > 0:
                self.commit(final=True)
            else:
                self.tempfile.unlink()
        
        self._closed = True


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
            raise NoDatasetFilesError(f"No files found in {self.dataset_dir} that match pattern: '**/*.zstd.parquet'")

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
            self.dataset_files = np.array_split(self.dataset_files, self.num_shards)[self.shard].tolist()

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

    # def get_dataframe(self) -> pl.LazyFrame:
    #     return pl.scan_parquet(self.dataset_files)

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
        return ds
    
    elif config.dataset_type == "parquet":
        logger.info("Loading parquet dataset with DatasetReader")
        return DatasetReader(config.dataset_path, shard=shard, num_shards=num_shards, **kwargs)
    else:
        raise ValueError(f"Invalid dataset type: {config.dataset_type}. Supported types are: 'hf', 'hf-disk', 'parquet'")
