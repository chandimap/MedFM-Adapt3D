"""Building and audit the local LIDC-IDRI cohort manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medfm_adapt3d.data.lidc_manifest import (
    build_audit_summary,
    build_lidc_manifest,
    write_audit_summary,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconcile LIDC-IDRI DICOM series and reader annotations "
            "into an auditable scan manifest."
        )
    )

    parser.add_argument(
        "--dicom-root",
        required=True,
        type=Path,
        help="Local root containing the downloaded TCIA LIDC-IDRI DICOM data.",
    )

    parser.add_argument(
        "--xml-root",
        required=True,
        type=Path,
        help="Local root containing the LIDC XML annotation files.",
    )

    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("data/manifests/lidc_idri_manifest.json"),
    )

    parser.add_argument(
        "--audit-out",
        type=Path,
        default=Path("data/manifests/lidc_idri_audit.json"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = build_lidc_manifest(
        dicom_root=args.dicom_root,
        xml_root=args.xml_root,
    )

    write_manifest(
        records,
        args.manifest_out,
    )

    write_audit_summary(
        records,
        args.audit_out,
    )

    summary = build_audit_summary(records)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()