import requests
import random
import yfinance as yf

def fetch_free_proxies():
    """
    Fetch a list of free HTTPS proxies from a public proxy provider.

    Returns:
        list: A list of proxy IPs as strings.
    """
    proxy_list_url = "https://www.proxy-list.download/api/v1/get?type=https"  # URL to fetch free proxies
    try:
        response = requests.get(proxy_list_url)  # Send a GET request to fetch the proxy list
        proxy_list = response.text.split("\r\n")[:-1]  # Split response into a list and remove the last empty element
        return proxy_list
    except:
        return []  # Return an empty list if there's an error (e.g., connection issues)

def test_proxy(proxy_list):
    """
    Test each proxy in the provided list to find one that works.

    Args:
        proxy_list (list): List of proxy IPs.

    Returns:
        str or None: A working proxy URL if found, otherwise None.
    """
    test_url = "https://query2.finance.yahoo.com/"  # Target URL to test if the proxy works
    headers = {"User-Agent": "Mozilla/5.0"}  # Mimic a real browser to avoid detection

    for proxy in proxy_list:
        try:
            proxy_dict = {"http": f"http://{proxy}", "https": f"https://{proxy}"}  # Format proxy for requests
            response_http =  yf.Ticker("GOOG", proxy["http"])  # Test proxy connection: http
            response_https =  yf.Ticker("GOOG", proxy["https"])  # Test proxy connection: http
        
            print(f"✅ Working Proxy: {proxy}")
            return proxy  # Return the first working proxy
        except:
            print(f"Failed proxy: ${proxy}")
            continue  # If a proxy fails, try the next one

    return None  # Return None if no working proxy is found

# Fetch the list of available free proxies
proxies = fetch_free_proxies()
print(proxies[:5])  # Print the first 5 proxies for debugging

# Attempt to find a working proxy
working_proxy = test_proxy(proxies)

# If a working proxy is found, use it with yfinance
if working_proxy:
    goog = yf.Ticker("GOOG", proxy=f"https://{working_proxy}")  # Use the working proxy
    print(goog.info)  # Print stock info for debugging
else:
    print("No valid proxies found!")  # Print an error message if no proxies work
