import re

# Read the backend.py file
with open(r'd:\COAL PROJECT AI CROP\server\backend.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace @app.get with @app.route(..., methods=["GET"])
content = re.sub(r'@app\.get\("([^"]+)"\)', r'@app.route("\1", methods=["GET"])', content)

# Replace @app.post with @app.route(..., methods=["POST"])
content = re.sub(r'@app\.post\("([^"]+)"\)', r'@app.route("\1", methods=["POST"])', content)

# Write back
with open(r'd:\COAL PROJECT AI CROP\server\backend.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Converted all @app.get and @app.post to @app.route for Flask compatibility")
print("   This ensures compatibility with Flask 1.x and 2.x")
