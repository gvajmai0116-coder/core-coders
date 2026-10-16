import sys
from pathlib import Path

CURRENT_PATH = Path(__file__).resolve().parent
if str(CURRENT_PATH) not in sys.path:
    sys.path.insert(0, str(CURRENT_PATH))

from tools.document_parser import parse_document

sample_bbraun_ocr_text = """
Organization: B. Braun Medical Inc.
Document Type: Discrepancy Notification (DN)
Document ID: SOP-MD-2024107
Version: 4.0
Effective Date: 2020-12-04
Proc Ref#: SOP-MD-2005725
Item #: E11040078
Batch #: 61828509
Machine/Line #: 574/688
Who is involved: J.D.
Where is the error: missing Data on R0pm
When did it happen: 5-24-2022
Other Details: 81998
Z7 DN #: 200125410
A7 DN #: 200125411
"""

def test_bbraun_parsing():
    parsed = parse_document(sample_bbraun_ocr_text)
    print("--- Test B. Braun Form Parsing Results ---")
    import json
    print(json.dumps(parsed, indent=2))
    
    fields = parsed.get("fields", {})
    assert parsed["organization"] == "B. Braun Medical Inc."
    assert fields["document_id"] == "SOP-MD-2024107"
    assert fields["version"] == "4.0"
    assert fields["effective_date"] == "2020-12-04"
    assert fields["proc_ref"] == "SOP-MD-2005725"
    assert fields["item_number"] == "E11040078"
    assert fields["batch_number"] == "61828509"
    assert fields["machine_line"] == "574/688"
    assert fields["who"] == "J.D."
    assert fields["where"] == "missing Data on R0pm"
    assert fields["when"] == "5-24-2022"
    assert fields["other_details"] == "81998"
    assert fields["z7_dn"] == "200125410"
    assert fields["a7_dn"] == "200125411"
    print("\n[SUCCESS] All B. Braun form field extractions verified successfully!")

if __name__ == "__main__":
    test_bbraun_parsing()
