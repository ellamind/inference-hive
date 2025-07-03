from pathlib import Path
from typing import Any, Dict

from loguru import logger
import polars as pl
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
    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        self.dataset_files = sorted(list(self.dataset_dir.glob("**/*.zstd.parquet")))
        logger.info(f"Found {len(self.dataset_files)} dataset files in {self.dataset_dir}")
        if len(self.dataset_files) == 0:
            raise NoDatasetFilesError(f"No files found in {self.dataset_dir} that match pattern: '**/*.zstd.parquet'")

        self.ds = pads.dataset(self.dataset_files)

    def __len__(self) -> int:
        return self.num_rows()

    def num_rows(self) -> int:
        return self.ds.count_rows()

    def num_files(self) -> int:
        return len(self.dataset_files)

    def get_dataset(self) -> pads.Dataset:
        return self.ds

    def get_schema(self) -> pa.Schema:
        return self.ds.schema

    def get_dataframe(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.dataset_files)
