import requests
import time

camera_ip = "192.168.8.105"
test_endpoints = [
    f"http://{camera_ip}/",
    f"http://{camera_ip}/capture",
    f"http://{camera_ip}/stream",
]

print("=" * 60)
print("Testing Camera Connection from Backend")
print("=" * 60)
print(f"Camera IP: {camera_ip}")
print()

for endpoint in test_endpoints:
    print(f"Testing: {endpoint}")
    try:
        start = time.time()
        r = requests.get(endpoint, timeout=10, verify=False)
        elapsed = time.time() - start
        print(f"  ✓ Status: {r.status_code}")
        print(f"  ✓ Response time: {elapsed:.2f}s")
        print(f"  ✓ Content-Type: {r.headers.get('content-type', 'N/A')}")
        if len(r.content) > 0:
            print(f"  ✓ Content size: {len(r.content)} bytes")
    except requests.exceptions.Timeout:
        print(f"  ✗ TIMEOUT (10s) - Camera not responding")
    except requests.exceptions.ConnectionError as e:
        print(f"  ✗ CONNECTION ERROR - {str(e)[:80]}")
    except Exception as e:
        print(f"  ✗ ERROR - {str(e)[:80]}")
    print()

print("=" * 60)
print("Next Steps:")
print("1. If all tests FAILED: Check if ESP32-CAM is online")
print("2. If some tests SUCCEEDED: Problem is specific to that endpoint")
print("3. If all tests PASSED: Problem is elsewhere")
print("=" * 60)
