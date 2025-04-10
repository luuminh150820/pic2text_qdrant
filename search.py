import requests
import json
import os
#from dotenv import load_dotenv

#pip install requests python-dotenv
#load_dotenv()

def google_search(query, api_key=None, cx=None, num=10):

    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    cx = cx or os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    
    # check key
    if not api_key:
        raise ValueError("Google API key is required.")
    if not cx:
        raise ValueError("Custom Search Engine ID is required.")
    
    # API URL
    url = "https://www.googleapis.com/customsearch/v1"
    
    params = {
        'q': query,
        'key': api_key,
        'cx': cx,
        'num': num
    }
    
    response = requests.get(url, params=params)

    #print(response.text)
    
    # Check request 
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

#format display
def display_search_results(results):

    if not results or 'items' not in results:
        print("No results found or API failed")
        return
        
    print(f"About {results.get('searchInformation', {}).get('totalResults', 'unknown')} results")
    print("-" * 80)
    
    for i, item in enumerate(results['items'], 1):
        print(f"{i}. {item['title']}")
        print(f"   URL: {item['link']}")
        print(f"   {item.get('snippet', 'No description available')}")
        print("-" * 80)

if __name__ == "__main__":
    
    search_query = "how to make fried chicken"
    
    
    # results = google_search(search_query,api_key,cx)
    results = google_search(search_query)
    
    if results:
        display_search_results(results)
    else:
        print("No search results.")
