#!/usr/bin/env python3
"""
Simple script to extract stock information from test.jpg
Usage: python extract_stock_simple.py
"""

import base64
import json
from openai import OpenAI
import os
import sys


def extract_stock_from_image(image_path="test.jpg"):
    """
    Extract stock information from image and return structured JSON
    
    Args:
        image_path: Path to the image file (default: test.jpg)
    
    Returns:
        Dictionary with extracted stock information
    """
    
    # Check if file exists
    if not os.path.exists(image_path):
        return {"error": f"Image file '{image_path}' not found"}
    
    # Check API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return {"error": "OPENAI_API_KEY environment variable not set"}
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Encode image to base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        # Create extraction prompt
        prompt = """
        Analyze this stock trading image and extract:
        1. 股票简称 (Stock Symbol/Name)
        2. 交易时间 (Trading Time)
        3. 交易额 (Trading Volume) - values > 100k shown as xxk or xxm
        
        Return as JSON:
        {
            "stocks": [
                {
                    "stock_symbol": "symbol",
                    "trading_time": "time",
                    "trading_volume": "volume"
                }
            ]
        }
        """
        
        # Call OpenAI Vision API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=1000
        )
        
        # Parse and return result
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Extract information
    result = extract_stock_from_image()
    
    # Output result
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # Save to file if successful
    if "error" not in result:
        with open("stock_info.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Saved to stock_info.json")
    else:
        print(f"\n❌ Error: {result['error']}", file=sys.stderr)
        sys.exit(1)
