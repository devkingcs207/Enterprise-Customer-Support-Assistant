
from pathlib import Path
import fitz

output_path = Path(__file__).parent / "sample_company_policy.pdf"

content = """
CUSTOMER SUPPORT POLICY

1. PRODUCT WARRANTY

Eligible electronic products include a 12-month limited warranty
beginning on the date of delivery. The warranty covers manufacturing
defects under normal use. It does not cover accidental damage,
liquid damage, unauthorized repairs, or normal wear and tear.

2. PRODUCT REPLACEMENT

Customers may request replacement for an item that arrives damaged
or defective within 7 calendar days of delivery. Requests must
include the order number and a clear description of the issue.
Replacement approval is not automatic and depends on review.

3. PAYMENT METHODS

Accepted payment methods include credit cards, debit cards,
and online bank transfers. Cash on delivery is not available.
Payment must be confirmed before an order is prepared for dispatch.

4. ORDER CANCELLATION

Customers may request cancellation before an order has been dispatched.
Once an order has been dispatched, it cannot be cancelled through
the standard cancellation process. Customers may contact support
to ask about the applicable return process.

5. CUSTOMER SUPPORT

Customers should contact support through the help form in their account.
They should include their order number and a clear description of the issue.
The standard support response target is within 2 business days.

TESTING NOTE

This document does not define international shipping, refund eligibility,
or refund processing times. The assistant should not infer these policies
from this document.
"""

pdf = fitz.open()
page = pdf.new_page()

page.insert_textbox(
    fitz.Rect(50, 50, 550, 750),
    content,
    fontsize=11,
    fontname="helv",
    lineheight=1.5
)

pdf.save(output_path)
pdf.close()

print(f"Company policy PDF created: {output_path}")
