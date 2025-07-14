#!/usr/bin/env python3
"""Test SQL Server syntax fixes"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from agents.sql_generator_agent import SQLGeneratorAgent

def test_limit_conversion():
    """Test the LIMIT to TOP conversion logic"""
    agent = SQLGeneratorAgent(None, None)  # We only need the cleaning function
    
    test_cases = [
        # Test case 1: LIMIT at end
        {
            "input": "SELECT * FROM dev.cliente ORDER BY customer_id LIMIT 10;",
            "expected_contains": "SELECT TOP 10",
            "expected_not_contains": "LIMIT"
        },
        # Test case 2: LIMIT without semicolon
        {
            "input": "SELECT * FROM dev.cliente LIMIT 5",
            "expected_contains": "SELECT TOP 5",
            "expected_not_contains": "LIMIT"
        },
        # Test case 3: Complex query with LIMIT
        {
            "input": """SELECT 
    mc.CEDI AS cedi_name,
    mc.CEDIid AS cedi_id,
    p.Producto AS product_name,
    SUM(s.IngresoNetoSImpuestos) AS total_revenue
FROM
    dev.segmentacion s
    JOIN dev.cliente_cedi cc ON s.customer_id = cc.customer_id
    JOIN dev.mercado mc ON cc.cedi_id = mc.CEDIid
    JOIN dev.producto p ON s.material_id = p.Material
WHERE
    cc.Region = 'Norte'
GROUP BY
    mc.CEDI, mc.CEDIid, p.Producto
ORDER BY
    total_revenue DESC
LIMIT 10""",
            "expected_contains": "SELECT TOP 10",
            "expected_not_contains": "LIMIT"
        }
    ]
    
    print("🧪 Testing LIMIT to TOP conversion...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['input'][:50]}...")
        
        # Apply the cleaning function
        cleaned_sql = agent._clean_sql_query(test_case["input"])
        
        print(f"Input:  {test_case['input']}")
        print(f"Output: {cleaned_sql}")
        
        # Check expected results
        if test_case["expected_contains"] in cleaned_sql:
            print(f"✅ Contains '{test_case['expected_contains']}'")
        else:
            print(f"❌ Missing '{test_case['expected_contains']}'")
            
        if test_case["expected_not_contains"] not in cleaned_sql:
            print(f"✅ Correctly removed '{test_case['expected_not_contains']}'")
        else:
            print(f"❌ Still contains '{test_case['expected_not_contains']}'")
        
        print("-" * 80)

if __name__ == "__main__":
    test_limit_conversion()
