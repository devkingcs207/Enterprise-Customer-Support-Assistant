
from pathlib import Path
import fitz

output_path = Path(__file__).parent / "sample_support_policy.pdf"

content = """
SAMPLE CUSTOMER SUPPORT POLICY

1. REFUND POLICY

Customers can request a refund within 30 days of purchase.
Refund requests must include the order number and proof of purchase.
Approved refunds are processed within 5 to 7 business days.

2. DELIVERY POLICY

Standard delivery usually takes 3 to 5 business days.
Customers can track their orders using the tracking link
provided in their shipping confirmation email.

3. ACCOUNT SUPPORT

Customers who forget their passwords can reset them through
the Forgot Password option on the login page.
For additional account assistance, customers can contact
the customer support team.
"""

pdf = fitz.open()
page = pdf.new_page()

page.insert_textbox(
    fitz.Rect(50, 50, 550, 750),
    content,
    fontsize=12,
    fontname="helv",
    lineheight=1.5
)

pdf.save(output_path)
pdf.close()

print(f"Sample PDF created: {output_path}")
