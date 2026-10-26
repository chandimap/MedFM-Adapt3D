"""Check the Public Representation Record and Claim Restrictions without Medical Data."""

from pathlib import Path

from medfm_adapt3d.foundation.provenance import claim_gate, load_model_record


def main() -> None:
    record = load_model_record(Path("configs/foundation/suprem_segresnet.yaml"))
    audit = claim_gate(record)
    if audit["real_held_out_test_enabled"] or not audit["allowed"]:
        raise RuntimeError("Unresolved provenance must permit engineering only.")
    for scope in ("clean_generalization", "clinical_readiness"):
        try:
            claim_gate(record, scope)
        except ValueError:
            continue
        raise RuntimeError(f"Unsupported claim was authorized: {scope}.")
    print("Frozen-representation identity and claim contract: PASS")
    print(f"Pinned model: {record.model_id}; patient-overlap status: {record.overlap_status}.")
    print("Official checkpoint loading is separately exercised by the explicit smoke script.")


if __name__ == "__main__":
    main()
