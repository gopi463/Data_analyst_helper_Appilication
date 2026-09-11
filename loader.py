"""
loader.py — Unified File Loader
AI Data Analyst Assistant

Supports: CSV, Excel (.xlsx/.xls), PDF
Returns a list of document dicts with text and metadata.
"""

import io
import pandas as pd
from pypdf import PdfReader
from pathlib import Path
from typing import Union


# ──────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────
def load_file(uploaded_file) -> dict:
    """
    Auto-detect file type and load content.

    Returns a dict:
    {
        "type":       "csv" | "excel" | "pdf",
        "filename":   str,
        "df":         pd.DataFrame | None,
        "documents":  [{"text": str, "file": str, "page": int, "type": str}],
        "sheet_names": list[str] | None,
        "error":      str | None
    }
    """
    filename = uploaded_file.name
    ext = Path(filename).suffix.lower()
    result = {
        "type": None,
        "filename": filename,
        "df": None,
        "documents": [],
        "sheet_names": None,
        "error": None,
    }

    try:
        if ext == ".csv":
            result.update(_load_csv(uploaded_file, filename))
        elif ext in (".xlsx", ".xls"):
            result.update(_load_excel(uploaded_file, filename))
        elif ext == ".pdf":
            result.update(_load_pdf(uploaded_file, filename))
        else:
            result["error"] = f"Unsupported file type: {ext}"
    except Exception as e:
        result["error"] = f"Error loading file: {str(e)}"

    return result


# ──────────────────────────────────────────
# CSV Loader
# ──────────────────────────────────────────
def _load_csv(uploaded_file, filename: str) -> dict:
    """Load a CSV file, auto-detect encoding."""
    content = uploaded_file.read()
    for enc in ["utf-8", "latin-1", "cp1252", "utf-16"]:
        try:
            df = pd.read_csv(io.BytesIO(content), encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        raise ValueError("Could not decode CSV file with any known encoding.")

    df = _clean_column_names(df)
    documents = _df_to_documents(df, filename, "csv")
    return {
        "type": "csv",
        "df": df,
        "documents": documents,
    }


# ──────────────────────────────────────────
# Excel Loader
# ──────────────────────────────────────────
def _load_excel(uploaded_file, filename: str) -> dict:
    """Load an Excel file. Uses the first sheet by default."""
    content = uploaded_file.read()
    xl = pd.ExcelFile(io.BytesIO(content))
    sheet_names = xl.sheet_names
    df = xl.parse(sheet_names[0])
    df = _clean_column_names(df)
    documents = _df_to_documents(df, filename, "excel")
    return {
        "type": "excel",
        "df": df,
        "documents": documents,
        "sheet_names": sheet_names,
    }


# ──────────────────────────────────────────
# PDF Loader
# ──────────────────────────────────────────
def _load_pdf(uploaded_file, filename: str) -> dict:
    """Load a PDF file and extract text page-by-page."""
    content = uploaded_file.read()
    reader = PdfReader(io.BytesIO(content))
    documents = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            documents.append({
                "text": text.strip(),
                "file": filename,
                "page": page_num,
                "type": "pdf",
            })
    return {
        "type": "pdf",
        "df": None,
        "documents": documents,
    }


# ──────────────────────────────────────────
# Multiple Files
# ──────────────────────────────────────────
def load_multiple_files(uploaded_files: list) -> tuple[list, list[dict], dict]:
    """
    Load multiple uploaded files.

    Returns:
        all_documents  : list of document dicts (for RAG)
        loaded_results : list of individual load result dicts
        combined_df    : first DataFrame found (for analytics)
    """
    all_documents = []
    loaded_results = []
    combined_df = None

    for f in uploaded_files:
        result = load_file(f)
        loaded_results.append(result)
        all_documents.extend(result.get("documents", []))
        if result.get("df") is not None and combined_df is None:
            combined_df = result["df"]

    return all_documents, loaded_results, combined_df


# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────
def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and normalize column names."""
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _df_to_documents(df: pd.DataFrame, filename: str, file_type: str, chunk_rows: int = 50) -> list[dict]:
    """
    Convert a DataFrame into text documents for RAG.
    Each chunk contains up to `chunk_rows` rows as a readable text block.
    """
    documents = []
    # 1. Include comprehensive schema and categorical catalog as the first document
    catalog_lines = [
        f"Dataset Catalog, Schema, Categories & Metadata for {filename}:",
        f"Total Rows: {len(df):,}, Total Columns: {len(df.columns)}",
        f"All Columns: {', '.join(df.columns.tolist())}",
        "\nCategorical Columns, Categories & Distinct Values (Ground Truth):",
    ]
    for col in df.columns:
        try:
            n_unique = df[col].nunique(dropna=True)
            if n_unique <= 50:
                unique_vals = [str(v) for v in df[col].dropna().unique().tolist()]
                try:
                    unique_vals.sort()
                except Exception:
                    pass
                catalog_lines.append(f"• Column '{col}' has {n_unique} unique categories/values: {', '.join(unique_vals)}")
            elif pd.api.types.is_numeric_dtype(df[col]):
                non_null = df[col].dropna()
                if not non_null.empty:
                    catalog_lines.append(f"• Column '{col}' (numeric): min={non_null.min()}, max={non_null.max()}, mean={non_null.mean():.2f}")
        except Exception:
            continue

    catalog_text = "\n".join(catalog_lines)
    documents.append({"text": catalog_text, "file": filename, "page": 0, "type": file_type})

    # Chunk data rows into text blocks
    for i in range(0, len(df), chunk_rows):
        chunk_df = df.iloc[i: i + chunk_rows]
        text = chunk_df.to_string(index=False)
        documents.append({
            "text": text,
            "file": filename,
            "page": i // chunk_rows + 1,
            "type": file_type,
        })
    return documents


def get_file_size_str(size_bytes: int) -> str:
    """Format file size as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 ** 2):.1f} MB"