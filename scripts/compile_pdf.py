"""
Construct a 20+ page academic PDF for the IDFS Capstone Mid-Term Evaluation Report.
"""

import os
import subprocess
import shutil

BASE_DIR = r"c:\Users\PAMDEM RAVI CHARAN\OneDrive\Desktop\SP-SP"
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
HTML_PATH = os.path.join(REPORTS_DIR, "midterm_report_expanded.html")
PDF_PATH = os.path.join(BASE_DIR, "midterm_report.pdf")
REPORTS_PDF_PATH = os.path.join(REPORTS_DIR, "midterm_report.pdf")

chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
temp_pdf = os.path.join(os.environ.get("TEMP", "C:\\Temp"), "midterm_report_compiled_exp.pdf")

args = [
    chrome_path,
    "--headless=new",
    "--disable-gpu",
    f"--print-to-pdf={temp_pdf}",
    f"file:///{HTML_PATH.replace(os.sep, '/')}"
]

print("Compiling PDF with Chrome...")
res = subprocess.run(args, capture_output=True, text=True)
if res.returncode == 0 and os.path.exists(temp_pdf):
    shutil.copy2(temp_pdf, PDF_PATH)
    shutil.copy2(temp_pdf, REPORTS_PDF_PATH)
    file_size = os.path.getsize(PDF_PATH)
    print(f"[SUCCESS] Generated PDF: {PDF_PATH}")
    print(f"          Size: {file_size / 1024:.2f} KB")
else:
    print(f"[ERROR] Failed: {res.stderr}")
