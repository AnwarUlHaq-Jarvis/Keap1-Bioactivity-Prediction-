import sys
import tempfile
import traceback
import logging
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st
from padelpy import padeldescriptor


# ============================================================
# 1. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Keap1 Bioactivity Prediction",
    page_icon="🧬",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_CANDIDATES = [
    BASE_DIR / "Keap1.pkl",
    BASE_DIR / "keap1.pkl",
]

FEATURE_LIST_PATH = BASE_DIR / "descriptor_list.csv"

FINGERPRINT_XML_CANDIDATES = [
    BASE_DIR / "PubchemFingerprinter.xml",
    BASE_DIR / "fingerprints_xml" / "PubchemFingerprinter.xml",
]

FINGERPRINT_XML = next(
    (
        path
        for path in FINGERPRINT_XML_CANDIDATES
        if path.is_file()
    ),
    FINGERPRINT_XML_CANDIDATES[0],
)

LOGO_PATH = BASE_DIR / "logo.jpg"

# PaDEL batch settings.
# Process compounds in manageable PaDEL batches.
INITIAL_BATCH_SIZE = 50

# Maximum number of compounds accepted in one prediction run.
MAX_COMPOUNDS_PER_RUN = 5000

# Configure application logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# 2. FIND MODEL FILE
# ============================================================

def find_model_file():
    """Find the trained model in the application directory."""

    for path in MODEL_CANDIDATES:
        if path.is_file():
            return path

    return None


# ============================================================
# 3. LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model(model_path, model_mtime):
    """Load the trained machine-learning model."""

    # Compatibility workaround for some NumPy 2.x pickles.
    # This does not guarantee compatibility with every pickle.
    if hasattr(np, "core"):
        sys.modules.setdefault("numpy._core", np.core)

    import pickle

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    if not hasattr(model, "predict"):
        raise TypeError(
            "The loaded model does not provide a predict() method."
        )

    return model


# ============================================================
# 4. READ SMILES FILE
# ============================================================

def read_smiles_file(uploaded_file):
    """
    Read SMILES records from CSV, TXT, or SMI files.

    Supported text examples:

        CCO Ethanol
        CC(=O)O Acetic_Acid

    Supported CSV example:

        SMILES,Molecule_Name
        CCO,Ethanol
        CC(=O)O,Acetic_Acid

    A file containing only SMILES is also accepted.
    """

    try:
        content = uploaded_file.getvalue().decode(
            "utf-8-sig",
            errors="replace",
        )

        if not content.strip():
            raise ValueError("The uploaded file is empty.")

        filename = uploaded_file.name.lower()

        if filename.endswith(".csv"):
            data = pd.read_csv(
                StringIO(content),
                dtype=str,
                comment="#",
                keep_default_na=False,
            )

            # Normalize header names for common CSV formats.
            normalized_columns = {
                str(column).strip().lower(): column
                for column in data.columns
            }

            smiles_header_options = {
                "smiles",
                "canonical_smiles",
                "isomeric_smiles",
                "canonical smiles",
                "molecule_smiles",
            }

            name_header_options = {
                "molecule_name",
                "name",
                "molecule",
                "compound_name",
                "compound",
                "id",
            }

            smiles_column = next(
                (
                    normalized_columns[column]
                    for column in smiles_header_options
                    if column in normalized_columns
                ),
                None,
            )

            name_column = next(
                (
                    normalized_columns[column]
                    for column in name_header_options
                    if column in normalized_columns
                ),
                None,
            )

            if smiles_column is not None:
                smiles_values = data[smiles_column].astype(str)

                if name_column is not None:
                    name_values = data[name_column].astype(str)
                else:
                    name_values = pd.Series(
                        [
                            f"Molecule_{i + 1}"
                            for i in range(len(data))
                        ],
                        index=data.index,
                    )

                data = pd.DataFrame({
                    "SMILES": smiles_values,
                    "Molecule_Name": name_values,
                })

            else:
                # Headerless CSV: assume column 1 is SMILES
                # and column 2, if present, is the molecule name.
                data = pd.read_csv(
                    StringIO(content),
                    header=None,
                    dtype=str,
                    comment="#",
                    keep_default_na=False,
                )

                if data.empty:
                    raise ValueError(
                        "The CSV file contains no molecular records."
                    )

                if data.shape[1] == 1:
                    data.columns = ["SMILES"]
                    data["Molecule_Name"] = [
                        f"Molecule_{i + 1}"
                        for i in range(len(data))
                    ]
                else:
                    data = data.iloc[:, :2].copy()
                    data.columns = ["SMILES", "Molecule_Name"]

        else:
            # Text-based .smi/.txt format:
            # SMILES followed by an optional molecule name.
            data = pd.read_csv(
                StringIO(content),
                sep=r"\s+",
                header=None,
                dtype=str,
                comment="#",
                engine="python",
                keep_default_na=False,
            )

            if data.empty:
                raise ValueError(
                    "The uploaded file contains no molecular records."
                )

            # Remove a recognized header if present.
            first_value = str(data.iloc[0, 0]).strip().lower()

            known_headers = {
                "smiles",
                "canonical_smiles",
                "isomeric_smiles",
                "canonical smiles",
                "molecule_smiles",
            }

            if first_value in known_headers:
                data = data.iloc[1:].reset_index(drop=True)

            if data.empty:
                raise ValueError(
                    "The file contains a header but no molecule records."
                )

            if data.shape[1] == 1:
                data.columns = ["SMILES"]
                data["Molecule_Name"] = [
                    f"Molecule_{i + 1}"
                    for i in range(len(data))
                ]
            else:
                data = data.iloc[:, :2].copy()
                data.columns = ["SMILES", "Molecule_Name"]

        # Clean values.
        data["SMILES"] = (
            data["SMILES"]
            .astype(str)
            .str.strip()
        )

        data["Molecule_Name"] = (
            data["Molecule_Name"]
            .astype(str)
            .str.strip()
        )

        # Remove empty SMILES entries.
        valid_smiles = (
            data["SMILES"].ne("")
            & data["SMILES"].str.lower().ne("nan")
            & data["SMILES"].str.lower().ne("none")
        )

        data = data.loc[valid_smiles].copy()

        if data.empty:
            raise ValueError(
                "No usable SMILES records remain after parsing."
            )

        # Replace empty molecule names.
        empty_names = (
            data["Molecule_Name"].eq("")
            | data["Molecule_Name"].str.lower().isin(
                {"nan", "none"}
            )
        )

        for index in data.index[empty_names]:
            data.loc[index, "Molecule_Name"] = (
                f"Molecule_{index + 1}"
            )

        data = data.reset_index(drop=True)

        # Track the original input order.
        data["_original_index"] = np.arange(len(data))

        return data

    except Exception as exc:
        raise ValueError(
            f"Could not read the SMILES file: {exc}"
        ) from exc


# ============================================================
# 5. CALCULATE DESCRIPTORS WITH PADEL
# ============================================================

def calculate_descriptors(input_data, working_dir):
    """
    Calculate PubChem fingerprints using PaDEL in small batches.

    If a batch fails, split it recursively to isolate molecules
    responsible for the failure.

    Returns:
        successful_data: input molecules processed successfully
        descriptor_data: numeric descriptor matrix
        failed_data: failed molecules and their error messages
    """

    if not FINGERPRINT_XML.is_file():
        raise FileNotFoundError(
            "PubchemFingerprinter.xml was not found.\n"
            "Checked locations:\n"
            f"1. {BASE_DIR / 'PubchemFingerprinter.xml'}\n"
            f"2. {BASE_DIR / 'fingerprints_xml' / 'PubchemFingerprinter.xml'}"
        )

    if input_data.empty:
        raise ValueError("There are no molecules to process.")

    required_columns = {
        "SMILES",
        "Molecule_Name",
    }

    missing_columns = required_columns - set(input_data.columns)

    if missing_columns:
        raise ValueError(
            f"Input data is missing columns: {sorted(missing_columns)}"
        )

    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    records = input_data.copy().reset_index(drop=True)

    # Use unique internal names regardless of user-provided names.
    records["_internal_name"] = [
        f"MOL_{i:08d}"
        for i in range(len(records))
    ]

    total = len(records)

    successful_rows = {}
    failed_records = []
    descriptor_columns = None

    completed = 0

    # --------------------------------------------------------
    # Progress indicators
    # --------------------------------------------------------

    progress_bar = st.progress(0.0)
    status = st.empty()

    status.info(
        f"Calculating fingerprints for {total} molecules..."
    )

    def update_progress(number_completed):
        nonlocal completed

        completed += number_completed

        progress_bar.progress(
            min(completed / total, 1.0)
        )

        status.info(
            f"Processed {completed}/{total} molecules."
        )

    # --------------------------------------------------------
    # Process one batch
    # --------------------------------------------------------

    def process_batch(batch):
        nonlocal descriptor_columns

        if batch.empty:
            return

        batch = batch.copy().reset_index(drop=True)

        expected_names = batch["_internal_name"].tolist()

        try:
            # Separate temporary directory for every attempt.
            with tempfile.TemporaryDirectory(
                prefix="padel_batch_",
                dir=str(working_dir),
            ) as temporary_directory:

                batch_dir = Path(temporary_directory)

                molecule_dir = batch_dir / "molecules"
                molecule_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                smiles_path = molecule_dir / "molecules.smi"
                output_path = batch_dir / "descriptors.csv"

                # PaDEL input:
                # SMILES<TAB>unique_molecule_name
                batch[
                    ["SMILES", "_internal_name"]
                ].to_csv(
                    smiles_path,
                    sep="\t",
                    index=False,
                    header=False,
                )

                # Do not pass maxruntime: its support varies
                # between padelpy versions. This call does not
                # impose a hard timeout on the Java process.
                padeldescriptor(
                    mol_dir=str(molecule_dir),
                    d_file=str(output_path),
                    descriptortypes=str(FINGERPRINT_XML),
                    detectaromaticity=True,
                    standardizenitro=True,
                    standardizetautomers=True,
                    removesalt=True,
                    fingerprints=True,
                    threads=1,
                )

                if not output_path.is_file():
                    raise RuntimeError(
                        "PaDEL did not create descriptors.csv."
                    )

                if output_path.stat().st_size == 0:
                    raise RuntimeError(
                        "PaDEL created an empty descriptor file."
                    )

                result = pd.read_csv(output_path)

                if result.empty:
                    raise RuntimeError(
                        "PaDEL returned no descriptor rows."
                    )

                result.columns = [
                    str(column).strip()
                    for column in result.columns
                ]

                # Remove duplicate output columns, if any.
                result = result.loc[
                    :,
                    ~result.columns.duplicated(),
                ]

                if "Name" not in result.columns:
                    raise ValueError(
                        "PaDEL output lacks the required Name column. "
                        f"Output columns: {result.columns.tolist()[:10]}"
                    )

                result["Name"] = (
                    result["Name"]
                    .astype(str)
                    .str.strip()
                )

                actual_names = result["Name"].tolist()

                if len(actual_names) != len(expected_names):
                    raise ValueError(
                        "Descriptor row count mismatch: "
                        f"expected {len(expected_names)}, "
                        f"received {len(actual_names)}."
                    )

                if len(set(actual_names)) != len(actual_names):
                    raise ValueError(
                        "PaDEL returned duplicate molecule names."
                    )

                if set(actual_names) != set(expected_names):
                    raise ValueError(
                        "PaDEL output names do not match input names.\n"
                        f"Expected: {expected_names}\n"
                        f"Received: {actual_names}"
                    )

                # Reorder rows to match the batch input order.
                result = result.set_index("Name")
                result = result.loc[expected_names]

                # Descriptor columns must be numeric.
                result = result.apply(
                    pd.to_numeric,
                    errors="coerce",
                )

                if result.shape[1] == 0:
                    raise ValueError(
                        "No numeric descriptor columns were returned."
                    )

                if result.isna().any().any():
                    bad_columns = result.columns[
                        result.isna().any()
                    ].tolist()

                    raise ValueError(
                        "Missing or non-numeric descriptor values "
                        f"were found in: {bad_columns[:10]}"
                    )

                values = result.to_numpy(dtype=float)

                if not np.isfinite(values).all():
                    raise ValueError(
                        "Descriptors contain infinite values."
                    )

                current_columns = result.columns.tolist()

                if descriptor_columns is None:
                    descriptor_columns = current_columns

                elif current_columns != descriptor_columns:
                    raise ValueError(
                        "Descriptor columns differ between batches."
                    )

                # Store rows by unique internal identifier.
                for internal_name in expected_names:
                    successful_rows[internal_name] = (
                        result.loc[internal_name].copy()
                    )

                update_progress(len(batch))

        except Exception as exc:

            error_message = (
                f"{type(exc).__name__}: {exc}"
            )

            logger.exception(
                "PaDEL batch failed. Molecules: %s",
                expected_names,
            )

            # Split failed batches to isolate problematic molecules.
            if len(batch) > 1:
                midpoint = len(batch) // 2

                process_batch(
                    batch.iloc[:midpoint].copy()
                )

                process_batch(
                    batch.iloc[midpoint:].copy()
                )

            else:
                row = batch.iloc[0]

                failed_records.append({
                    "Molecule_Name": row["Molecule_Name"],
                    "SMILES": row["SMILES"],
                    "Error": error_message,
                })

                update_progress(1)

    # --------------------------------------------------------
    # Process initial batches
    # --------------------------------------------------------

    batch_size = max(
        1,
        int(INITIAL_BATCH_SIZE),
    )

    for start in range(0, total, batch_size):
        batch = records.iloc[
            start:start + batch_size
        ].copy()

        process_batch(batch)

    progress_bar.progress(1.0)

    # --------------------------------------------------------
    # Assemble failed results
    # --------------------------------------------------------

    failed_data = pd.DataFrame(
        failed_records,
        columns=[
            "Molecule_Name",
            "SMILES",
            "Error",
        ],
    )

    # --------------------------------------------------------
    # Handle complete failure
    # --------------------------------------------------------

    if not successful_rows:
        status.error(
            "PaDEL failed to process every molecule."
        )

        if not failed_data.empty:
            st.error("PaDEL error details:")

            st.dataframe(
                failed_data,
                use_container_width=True,
            )

            st.download_button(
                label="Download PaDEL Error Report",
                data=failed_data.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="padel_error_report.csv",
                mime="text/csv",
                key="padel_all_failed_download",
            )

            first_error = failed_data.iloc[0]["Error"]

            raise RuntimeError(
                "No molecules were processed successfully. "
                f"First PaDEL error: {first_error}"
            )

        raise RuntimeError(
            "No molecules were processed and no error details "
            "were recorded."
        )

    # --------------------------------------------------------
    # Preserve original input order
    # --------------------------------------------------------

    successful_internal_names = [
        name
        for name in records["_internal_name"]
        if name in successful_rows
    ]

    successful_data = (
        records[
            records["_internal_name"].isin(
                successful_internal_names
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    descriptor_data = pd.DataFrame(
        [
            successful_rows[name]
            for name in successful_internal_names
        ],
        columns=descriptor_columns,
    ).reset_index(drop=True)

    status.success(
        f"PaDEL finished: {len(successful_data)} successful; "
        f"{len(failed_data)} failed; {total} total."
    )

    return successful_data, descriptor_data, failed_data


# ============================================================
# 6. LOAD SELECTED FEATURE LIST
# ============================================================

def load_feature_list(descriptor_columns):
    """Load selected descriptor names from descriptor_list.csv."""

    if not FEATURE_LIST_PATH.is_file():
        raise FileNotFoundError(
            f"Missing feature-selection file:\n{FEATURE_LIST_PATH}"
        )

    feature_table = pd.read_csv(
        FEATURE_LIST_PATH,
        header=None,
        dtype=str,
        keep_default_na=False,
    )

    raw_features = (
        feature_table
        .fillna("")
        .astype(str)
        .values
        .flatten()
        .tolist()
    )

    descriptor_set = set(descriptor_columns)

    selected_features = []
    seen = set()

    ignored_labels = {
        "feature",
        "features",
        "descriptor",
        "descriptors",
        "feature_name",
        "descriptor_name",
        "index",
    }

    for feature in raw_features:
        feature = feature.strip()

        if not feature:
            continue

        if feature.lower() in ignored_labels:
            continue

        if (
            feature in descriptor_set
            and feature not in seen
        ):
            selected_features.append(feature)
            seen.add(feature)

    if not selected_features:
        raise ValueError(
            "None of the feature names in descriptor_list.csv "
            "match the calculated PaDEL descriptors. Check that "
            "the file contains the original model feature names."
        )

    return selected_features


# ============================================================
# 7. DETERMINE EXPECTED MODEL FEATURE COUNT
# ============================================================

def get_expected_feature_count(model):
    """Determine the number of input features expected by the model."""

    count = getattr(model, "n_features_in_", None)

    if count is not None:
        return int(count)

    importances = getattr(
        model,
        "feature_importances_",
        None,
    )

    if importances is not None:
        return len(importances)

    coefficients = getattr(model, "coef_", None)

    if coefficients is not None:
        coefficients = np.asarray(coefficients)

        if coefficients.ndim == 1:
            return coefficients.shape[0]

        if coefficients.ndim == 2:
            return coefficients.shape[1]

    # Look inside a scikit-learn pipeline.
    steps = getattr(model, "steps", None)

    if steps:
        for _, estimator in reversed(steps):
            count = getattr(
                estimator,
                "n_features_in_",
                None,
            )

            if count is not None:
                return int(count)

            importances = getattr(
                estimator,
                "feature_importances_",
                None,
            )

            if importances is not None:
                return len(importances)

            coefficients = getattr(
                estimator,
                "coef_",
                None,
            )

            if coefficients is not None:
                coefficients = np.asarray(coefficients)

                if coefficients.ndim == 1:
                    return coefficients.shape[0]

                if coefficients.ndim == 2:
                    return coefficients.shape[1]

    return None


# ============================================================
# 8. ALIGN FEATURES WITH MODEL
# ============================================================

def align_features(
    descriptors,
    model,
    selected_features,
):
    """Select model features in the correct order."""

    model_feature_names = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if model_feature_names is not None:
        required_features = [
            str(feature)
            for feature in model_feature_names
        ]

        missing = [
            feature
            for feature in required_features
            if feature not in descriptors.columns
        ]

        if missing:
            raise ValueError(
                "The model requires descriptor names missing "
                "from the calculated matrix:\n"
                f"{missing[:20]}"
            )

        features = descriptors[
            required_features
        ].copy()

    else:
        missing = [
            feature
            for feature in selected_features
            if feature not in descriptors.columns
        ]

        if missing:
            raise ValueError(
                "Selected features are missing from the descriptor "
                f"matrix: {missing[:20]}"
            )

        features = descriptors[
            selected_features
        ].copy()

    expected_count = get_expected_feature_count(model)
    actual_count = features.shape[1]

    if (
        expected_count is not None
        and actual_count != expected_count
    ):
        raise ValueError(
            "Feature-count mismatch.\n\n"
            f"Model expects: {expected_count}\n"
            f"Application supplies: {actual_count}\n\n"
            "Restore the exact feature list used during training. "
            "The application will not add or remove features automatically."
        )

    if features.empty:
        raise ValueError(
            "The selected feature matrix is empty."
        )

    values = features.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "The feature matrix contains NaN or infinite values."
        )

    return features


# ============================================================
# 9. GENERATE PREDICTIONS
# ============================================================

def predict_bioactivity(model, features):
    """Generate predictions and validate their output."""

    try:
        predictions = model.predict(features)

        predictions = np.asarray(
            predictions
        ).reshape(-1)

        if len(predictions) != len(features):
            raise ValueError(
                "The number of predictions does not match "
                "the number of molecules."
            )

        predictions = predictions.astype(float)

        if not np.isfinite(predictions).all():
            raise ValueError(
                "The model returned non-finite prediction values."
            )

        return predictions

    except Exception as exc:
        raise RuntimeError(
            f"Model prediction failed: {exc}"
        ) from exc


# ============================================================
# 10. USER INTERFACE
# ============================================================

if LOGO_PATH.is_file():
    st.image(
        str(LOGO_PATH),
        use_column_width=True,
    )

st.title("Keap1 Bioactivity Prediction App")
st.subheader("Machine-Learning Prediction Engine")

st.markdown(
    """
    Upload a file containing molecular SMILES strings to calculate
    PubChem fingerprints and predict bioactivity using the trained
    Keap1 machine-learning model.

    **Target:** Keap1 (Kelch-like ECH-associated protein 1)

    Molecules that fail PaDEL processing will be reported separately.
    Successfully processed molecules can still proceed to prediction.
    """
)

st.divider()

st.sidebar.header("1. Input Dataset")

uploaded_files = st.sidebar.file_uploader(
    "Upload SMILES files (up to 5,000 compounds total)",
    type=["txt", "smi", "csv"],
    accept_multiple_files=True,
)

model_path = find_model_file()

if model_path is None:
    st.error(
        "Model file not found. Place Keap1.pkl or keap1.pkl "
        "in the same directory as app.py."
    )

if not FEATURE_LIST_PATH.is_file():
    st.error(
        "descriptor_list.csv is missing from the application directory."
    )

if not FINGERPRINT_XML.is_file():
    st.error(
        "PubchemFingerprinter.xml was not found. Place it in the "
        "project root or in fingerprints_xml/."
    )

with st.sidebar.expander("Required application files"):
    st.markdown(
        """
        - `app.py`
        - `Keap1.pkl` or `keap1.pkl`
        - `descriptor_list.csv`
        - `PubchemFingerprinter.xml` or
          `fingerprints_xml/PubchemFingerprinter.xml`
        - `requirements.txt`
        - `packages.txt`
        """
    )

predict_button = st.sidebar.button(
    "Predict Bioactivity",
    type="primary",
    use_container_width=True,
)


# ============================================================
# 11. MAIN EXECUTION
# ============================================================

# ============================================================
# 11. MAIN EXECUTION
# ============================================================

if predict_button:

    if not uploaded_files:
        st.sidebar.error("Please upload at least one SMILES file.")

    elif model_path is None:
        st.error("Cannot continue because the model file is missing.")

    elif not FEATURE_LIST_PATH.is_file():
        st.error("Cannot continue because descriptor_list.csv is missing.")

    elif not FINGERPRINT_XML.is_file():
        st.error("Cannot continue because the fingerprint XML file is missing.")

    else:

        try:
            # ------------------------------------------------
            # STEP A: Read and combine uploaded files
            # ------------------------------------------------

            input_frames = []
            total_input = 0

            with st.spinner("Reading uploaded molecular files..."):

                for uploaded_file in uploaded_files:

                    file_data = read_smiles_file(uploaded_file)

                    input_frames.append(file_data)
                    total_input += len(file_data)

                    if total_input > MAX_COMPOUNDS_PER_RUN:
                        raise ValueError(
                            f"You uploaded {total_input:,} compounds. "
                            f"The maximum allowed per run is "
                            f"{MAX_COMPOUNDS_PER_RUN:,}."
                        )

            input_data = pd.concat(
                input_frames,
                ignore_index=True,
            )

            # Reset the original input order across all files.
            input_data["_original_index"] = np.arange(
                len(input_data)
            )

            st.subheader("Input Dataset")

            st.write(
                f"Uploaded files: {len(uploaded_files)}"
            )

            st.write(
                f"Total input compounds: {len(input_data):,}"
            )

            st.dataframe(
                input_data[
                    ["SMILES", "Molecule_Name"]
                ].head(20),
                use_container_width=True,
            )

            # ------------------------------------------------
            # STEP B: Calculate PaDEL fingerprints
            # ------------------------------------------------

            st.subheader("PaDEL Fingerprint Calculation")

            with tempfile.TemporaryDirectory(
                prefix="keap1_prediction_"
            ) as temp_dir:

                working_dir = Path(temp_dir)

                successful_data, descriptors, failed_data = (
                    calculate_descriptors(
                        input_data,
                        working_dir,
                    )
                )

                # --------------------------------------------
                # Report failed molecules
                # --------------------------------------------

                if not failed_data.empty:

                    st.warning(
                        f"{len(failed_data):,} compounds failed "
                        "PaDEL processing. The remaining compounds "
                        "will continue to prediction."
                    )

                    st.dataframe(
                        failed_data.head(20),
                        use_container_width=True,
                    )

                    st.download_button(
                        label="Download Failed Molecules Report",
                        data=failed_data.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="keap1_failed_molecules.csv",
                        mime="text/csv",
                        key="failed_molecules_download",
                    )

                # --------------------------------------------
                # Show descriptor summary, not the full matrix
                # --------------------------------------------

                st.subheader("Calculated Fingerprints")

                st.write(
                    f"Successfully processed: "
                    f"{len(successful_data):,}"
                )

                st.write(
                    f"Failed: {len(failed_data):,}"
                )

                st.write(
                    f"Descriptor matrix shape: {descriptors.shape}"
                )

                st.dataframe(
                    descriptors.head(10),
                    use_container_width=True,
                )

                # --------------------------------------------
                # STEP C: Load selected features
                # --------------------------------------------

                selected_features = load_feature_list(
                    descriptors.columns
                )

                st.subheader("Model Feature Selection")

                st.write(
                    f"Selected feature count: "
                    f"{len(selected_features)}"
                )

                # --------------------------------------------
                # STEP D: Load trained model
                # --------------------------------------------

                with st.spinner("Loading trained Keap1 model..."):

                    model = load_model(
                        str(model_path),
                        model_path.stat().st_mtime,
                    )

                st.write(
                    f"Model loaded: {model_path.name}"
                )

                expected_count = get_expected_feature_count(
                    model
                )

                st.write(
                    f"Model expected feature count: "
                    f"{expected_count}"
                )

                # --------------------------------------------
                # STEP E: Align model features
                # --------------------------------------------

                features = align_features(
                    descriptors,
                    model,
                    selected_features,
                )

                if features.shape[1] != 179:
                    raise ValueError(
                        "Incorrect number of selected features. "
                        f"Expected 179, received {features.shape[1]}."
                    )

                st.success(
                    f"Feature matrix validated: "
                    f"{features.shape[0]:,} molecules × "
                    f"{features.shape[1]} features."
                )

                st.dataframe(
                    features.head(10),
                    use_container_width=True,
                )

                # --------------------------------------------
                # STEP F: Generate predictions
                # --------------------------------------------

                with st.spinner(
                    f"Predicting bioactivity for "
                    f"{len(features):,} compounds..."
                ):

                    predictions = predict_bioactivity(
                        model,
                        features,
                    )

                # --------------------------------------------
                # STEP G: Construct prediction results
                # --------------------------------------------

                results = pd.DataFrame({
                    "Molecule_Name": successful_data[
                        "Molecule_Name"
                    ].to_numpy(),

                    "SMILES": successful_data[
                        "SMILES"
                    ].to_numpy(),

                    "Predicted_pIC50": predictions,
                })

                st.success(
                    f"Prediction completed successfully for "
                    f"{len(results):,} compounds."
                )

                # --------------------------------------------
                # STEP H: Display and download results
                # --------------------------------------------

                st.subheader("Keap1 Bioactivity Predictions")

                st.dataframe(
                    results,
                    use_container_width=True,
                )

                st.download_button(
                    label="Download Prediction Results",
                    data=results.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="keap1_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="prediction_results_download",
                )

                st.download_button(
                    label="Download Selected Feature Matrix",
                    data=features.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="keap1_selected_features.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="feature_matrix_download",
                )

                st.info(
                    "Predicted pIC50 values are computational estimates "
                    "from the trained Keap1 model, not experimental "
                    "measurements. Interpret them within the chemical "
                    "and activity range represented in the training data."
                )

        except Exception as exc:

            st.error(
                "The prediction pipeline encountered an error."
            )

            st.code(
                f"{type(exc).__name__}: {exc}",
                language="text",
            )

            with st.expander("Technical details"):

                st.code(
                    traceback.format_exc(),
                    language="python",
                )

else:

    st.info(
        "Upload one or more SMILES files in the sidebar, "
        "then click Predict Bioactivity. "
        "The combined input limit is 5,000 compounds."
    )