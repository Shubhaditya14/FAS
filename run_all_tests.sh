#!/bin/bash

# FAS System - Run All Tests
# This script runs all test files and saves output to test_results.txt

echo "=========================================="
echo "FAS System - Running All Tests"
echo "=========================================="
echo ""

# Remove old results file
rm -f test_results.txt

# Create results file with header
cat > test_results.txt << 'EOF'
========================================
FAS SYSTEM TEST RESULTS
========================================
Date: $(date)
========================================

EOF

# Activate virtual environment
source venv/bin/activate

# Test 1: Architecture
echo "Running Test 1: Model Architecture..."
echo "" >> test_results.txt
echo "========================================" >> test_results.txt
echo "TEST 1: MODEL ARCHITECTURE" >> test_results.txt
echo "========================================" >> test_results.txt
python test_architecture.py >> test_results.txt 2>&1
echo "" >> test_results.txt

# Test 2: Weight Loading
echo "Running Test 2: Pretrained Weight Loading..."
echo "" >> test_results.txt
echo "========================================" >> test_results.txt
echo "TEST 2: PRETRAINED WEIGHT LOADING" >> test_results.txt
echo "========================================" >> test_results.txt
python test_weight_loading.py >> test_results.txt 2>&1
echo "" >> test_results.txt

# Test 3: Preprocessing
echo "Running Test 3: Preprocessing Pipeline..."
echo "" >> test_results.txt
echo "========================================" >> test_results.txt
echo "TEST 3: PREPROCESSING PIPELINE" >> test_results.txt
echo "========================================" >> test_results.txt
python test_preprocessing.py >> test_results.txt 2>&1
echo "" >> test_results.txt

# Test 4: Datasets
echo "Running Test 4: Dataset Loading..."
echo "" >> test_results.txt
echo "========================================" >> test_results.txt
echo "TEST 4: DATASET LOADING" >> test_results.txt
echo "========================================" >> test_results.txt
python test_datasets.py >> test_results.txt 2>&1
echo "" >> test_results.txt

# Test 5: Inference
echo "Running Test 5: Single Image Inference..."
echo "" >> test_results.txt
echo "========================================" >> test_results.txt
echo "TEST 5: SINGLE IMAGE INFERENCE" >> test_results.txt
echo "========================================" >> test_results.txt
python test_inference.py >> test_results.txt 2>&1
echo "" >> test_results.txt

# Test 6: Evaluation
echo "Running Test 6: Full Dataset Evaluation..."
echo "" >> test_results.txt
echo "========================================" >> test_results.txt
echo "TEST 6: FULL DATASET EVALUATION" >> test_results.txt
echo "========================================" >> test_results.txt
python test_evaluation.py >> test_results.txt 2>&1
echo "" >> test_results.txt

# Footer
cat >> test_results.txt << 'EOF'

========================================
ALL TESTS COMPLETED
========================================
EOF

echo ""
echo "=========================================="
echo "All tests completed!"
echo "Results saved to test_results.txt"
echo "=========================================="
echo ""

# Display results
cat test_results.txt
