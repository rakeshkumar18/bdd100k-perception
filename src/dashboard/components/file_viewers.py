"""Reusable file display helpers."""

from pathlib import Path

import pandas as pd
import streamlit as st


def show_image(
    image_name: str,
    caption: str,
    image_dir: Path,
) -> None:
    """Display image from directory."""

    image_path = (
        image_dir / image_name
    )

    if image_path.exists():

        st.image(
            str(image_path),
            caption=caption,
            width="stretch",
        )

    else:

        st.warning(
            f"Missing image: "
            f"{image_path}"
        )


def show_image_if_exists(
    image_path: Path,
    caption: str,
) -> None:
    """Display image if available."""

    if image_path.exists():

        st.image(
            str(image_path),
            caption=caption,
            width="stretch",
        )

    else:

        st.warning(
            f"Missing artifact: "
            f"{image_path.name}"
        )


def show_table(
    csv_name: str,
    title: str,
    table_dir: Path,
) -> None:
    """Display CSV table."""

    csv_path = (
        table_dir / csv_name
    )

    if csv_path.exists():

        st.markdown(
            f"#### {title}"
        )

        st.dataframe(
            pd.read_csv(csv_path),
            width="stretch",
        )

    else:

        st.info(
            f"Missing table: "
            f"{csv_path}"
        )