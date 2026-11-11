# Capstone data packet — Fall 2026

This packet contains 14 fixed, prepared datasets. Loading requires no network,
account, credentials, TensorFlow, or system HDF5 installation. Python 3.12 is the
tested interpreter. Install the two dependencies once, then validate offline:

```bash
python -m pip install -r requirements.txt
python capstone_data.py
```

Open `capstone_data_loader.ipynb` in your existing Jupyter/Colab environment, or
import the loader from a script in this directory:

```python
from capstone_data import load_dataset, chronological_split
df = load_dataset("jena")
train, validation, test = chronological_split(df)
print(df.attrs["capstone"])
```

For Colab, upload and extract the entire ZIP, then change to its
`capstone_support` directory before importing. A notebook uploaded alone does
not include the snapshots. Model packages are a separate choice; these pinned
dependencies only load and validate data.

## Contents and provenance

- `snapshots/`: 14 compressed numeric CSVs plus PEMS location/distance metadata.
- `data_manifest.json`: exact source URLs, raw hashes, prepared hashes, coverage,
  columns, units, preparation rules, targets, horizons, and missing-value counts.
- `capstone_data.py`: the single offline loader used by the handout and notebook.
- `capstone_data_loader.ipynb`: executed examples for all 14 IDs; its final table
  is computed by validation, not a manually written readiness claim.
- `validation.json`: machine-generated results from the release checks.

The notebook lists all IDs. Metadata returned in `df.attrs['capstone']` describes
your dataset. Cite its `source_page`, original data creators, and this course
preparation. UCI datasets identify their CC BY 4.0 attribution terms on their
source pages; MIT-BIH identifies its ODC attribution terms on PhysioNet. Other
providers retain their respective terms. The snapshots are observed source data,
not synthetic substitutes. Files and checksums are frozen for this course release.

M5 is replaced by UCI Online Retail; Rossmann by Capital Bikeshare; the Wikipedia
competition by an official Wikimedia API extract; HAR by UCI appliance/sensor
data. The solar data come from Anik Kannal's Kaggle plant dataset, not NREL.
The solar generation and weather files were already available locally and were
used to build the included snapshot; no student Kaggle login is needed.

## Timing and missing values

Missing observations remain missing. A regular index does not mean every value
was observed. Do not remove missing rows and then treat the remaining rows as a
regular time series. Fit imputation, scaling, PCA, and feature selection within
each training window, using information available at the origin. Do not impute
held-out targets for scoring. Use the same observed evaluation targets for all
models and disclose the scored count.

`chronological_split` returns first-60%, next-20%, final-20% time blocks. Split
before constructing examples. Every example's entire target window must remain
inside its assigned block. Validation chooses model settings; final test follows
a frozen evaluation procedure. Earlier observed history may be used as context.
Aggregates become available after their interval ends, even when the index labels
the interval start. Future measured weather/sensors/other-series values are not
automatically future-known predictors. See the handout's evaluation contract.

## Instructor rebuilding (repository only)

The repository additionally contains `build_data.py`, `requirements-build.txt`,
raw Jena/solar inputs, `test_data.py`, and `make_packet.py`; these maintenance
files are not required by students. From the repository root:

```bash
python project/data_check/build_data.py --cache /tmp/math546-capstone-raw
python project/data_check/test_data.py
python project/data_check/build_notebook.py
python project/data_check/make_packet.py
```

The builder downloads only documented public source files and raises errors on
failed/empty inputs. Rebuilding is an explicit maintenance operation: providers
can revise data and the new manifest/packet must be reviewed and revalidated.
Cached raw files are hashed. A failed rebuild may leave partially refreshed
snapshots; do not release until every check and notebook execution succeeds.

If local TLS configuration requires an institution-approved CA bundle, set
`REQUESTS_CA_BUNDLE` and `SSL_CERT_FILE` to that bundle for the instructor build.
Do not disable certificate verification. Student loading never makes requests.
