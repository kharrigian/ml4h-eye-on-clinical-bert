#!/bin/bash

echo "Placing Synthetic Note Files in a CSV"
python -u data/resources/synthetic-notes/convert2csv.py

echo "Placing Synthetic Note Files in a JSON"
python -u data/resources/synthetic-notes/convert2json.py


