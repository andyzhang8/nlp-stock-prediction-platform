from PyPDF2 import PdfReader
import csv

# Path to the uploaded PDF file
pdf_path = 'data\\ru3000_membershiplist_20220624.pdf'
csv_output_path = 'data\\Russell3000_Companies.csv'

# Read the PDF file
reader = PdfReader(pdf_path)
text = ""

# Extract text from each page
for page in reader.pages:
    text += page.extract_text()

# Process text to extract company names and tickers
lines = text.split("\n")

tickers = []

for line in lines:
    if "Company" in line and "Ticker" in line:
        continue  # Skip header rows
    line = line.split(" ")
    # Ensure line[0] is all uppercase letters and nothing else
    if len(line) == 1 and (line[0].isalpha() and line[0].isupper()) and len(line[0]) <= 5:
        tickers.append(line)
    

# Save to CSV file
with open(csv_output_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Company"])
    writer.writerows(tickers)

