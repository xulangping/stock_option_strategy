import base64
import json
from openai import OpenAI
from typing import Dict, Any
import os

def encode_image(image_path: str) -> str:
    """
    Encode image to base64 string
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_stock_info(image_path: str, api_key: str = None) -> Dict[str, Any]:
    """
    Extract stock information from image using OpenAI Vision API
    
    Args:
        image_path: Path to the stock image
        api_key: OpenAI API key (optional, will use environment variable if not provided)
    
    Returns:
        Dictionary containing extracted stock information
    """
    
    # Initialize OpenAI client
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        # Use environment variable OPENAI_API_KEY
        client = OpenAI()
    
    # Encode the image
    base64_image = encode_image(image_path)
    
    # Create the prompt for structured extraction
    prompt = """
    Please analyze this stock trading image and extract the following information:
    1. Stock Symbol/Name (股票简称)
    2. Trading Time (交易时间)
    3. Trading Volume (交易额) - Look for values greater than 100k, formatted as xxk or xxm
    
    Return the information in the following JSON format:
    {
        "stocks": [
            {
                "stock_symbol": "stock name or symbol",
                "trading_time": "trading time",
                "trading_volume": "volume (keep original format like 123k or 1.2m)"
            }
        ]
    }
    
    If there are multiple stocks in the image, include all of them in the array.
    Only extract the information that is clearly visible in the image.
    For trading volume, focus on the column with values greater than 100k (shown as xxk or xxm).
    """
    
    try:
        # Make API call using the chat completions endpoint with vision
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or use "gpt-4-turbo" for better accuracy
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"  # Use high detail for better text recognition
                            }
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},  # Request JSON response
            max_tokens=1000
        )
        
        # Parse the response
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {"error": str(e)}


def main():
    """
    Main function to extract stock information from test.jpg
    """
    # Image path
    image_path = "test.jpg"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    # Get API key from environment variable or set it here
    # api_key = "your-api-key-here"  # Uncomment and set your API key if not using environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OpenAI API key not found!")
        print("Please set the OPENAI_API_KEY environment variable or provide it in the code.")
        return
    
    print(f"Extracting stock information from {image_path}...")
    
    # Extract information
    result = extract_stock_info(image_path, api_key)
    
    # Print results
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("\nExtracted Stock Information:")
        print("=" * 50)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Save to JSON file
        output_file = "extracted_stock_info.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_file}")
        
        # Print summary
        if "stocks" in result:
            print(f"\nTotal stocks extracted: {len(result['stocks'])}")
            for i, stock in enumerate(result['stocks'], 1):
                print(f"\nStock {i}:")
                print(f"  Symbol: {stock.get('stock_symbol', 'N/A')}")
                print(f"  Trading Time: {stock.get('trading_time', 'N/A')}")
                print(f"  Trading Volume: {stock.get('trading_volume', 'N/A')}")


if __name__ == "__main__":
    main()
