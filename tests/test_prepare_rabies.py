from fixrabies.interfaces.prepare_rabies import prepare_rabies


def test_prepare_rabies():
    scans = prepare_rabies("~/test/preprocessing")

    assert isinstance(scans, dict)
    assert len(scans) > 0

    for scan_id, scan in scans.items():
        print(f"\n{scan_id}")
        for key, value in scan.items():
            print(f"  {key}: {value}")

        assert scan["bold_file"] is not None
