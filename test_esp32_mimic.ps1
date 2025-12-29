$baseUrl = "https://ai-crop-disease-detection.onrender.com"
# $baseUrl = "http://127.0.0.1:5000" # Uncomment to test local server

# ==========================================
# 1. Test Sensor Upload (Mimic ESP32 String)
# ==========================================
Write-Host "--- 1. Testing Sensor Upload (ESP32 Mimic) ---"
# Exact format from sketch_nov19a.ino:
# snprintf(json, ..., "{\"temperature\":%.1f,\"humidity\":%.1f,\"soil\":%d,\"rain\":%d,\"light\":%.1f}", ...)
$jsonPayload = '{"temperature":25.5,"humidity":60.0,"soil":2048,"rain":4095,"light":120.5}'

try {
    $response = Invoke-RestMethod -Uri "$baseUrl/sensors/store" -Method Post -Body $jsonPayload -ContentType "application/json"
    Write-Host "✅ Sensor Upload Success: $($response | ConvertTo-Json -Depth 1)" -ForegroundColor Green
} catch {
    Write-Host "❌ Sensor Upload Failed: $_" -ForegroundColor Red
    try {
        $stream = $_.Exception.Response.GetResponseStream()
        $reader = New-Object System.IO.StreamReader($stream)
        Write-Host "   Server Response: $($reader.ReadToEnd())" -ForegroundColor Yellow
    } catch {}
}

# ==========================================
# 2. Test Camera Upload (Mimic ESP32-CAM)
# ==========================================
Write-Host "`n--- 2. Testing Camera Upload (ESP32-CAM Mimic) ---"
$localImagePath = "d:\COAL PROJECT AI CROP\test_cam.jpg"

# Create a dummy image if not exists
if (-not (Test-Path $localImagePath)) {
    python -c "from PIL import Image; img = Image.new('RGB', (320, 240), color=(100, 100, 255)); img.save(r'$localImagePath', 'JPEG')"
}

try {
    # Read file bytes
    $bytes = [System.IO.File]::ReadAllBytes($localImagePath)
    
    # Send as raw binary with image/jpeg content type (mimicking ESP32)
    $req = [System.Net.HttpWebRequest]::Create("$baseUrl/camera/upload")
    $req.Method = "POST"
    $req.ContentType = "image/jpeg"
    $req.ContentLength = $bytes.Length
    $stream = $req.GetRequestStream()
    $stream.Write($bytes, 0, $bytes.Length)
    $stream.Close()

    $resp = $req.GetResponse()
    $reader = New-Object System.IO.StreamReader($resp.GetResponseStream())
    $result = $reader.ReadToEnd()
    Write-Host "✅ Camera Upload Success: $result" -ForegroundColor Green
} catch {
    Write-Host "❌ Camera Upload Failed: $_" -ForegroundColor Red
    try {
        $stream = $_.Exception.Response.GetResponseStream()
        $reader = New-Object System.IO.StreamReader($stream)
        Write-Host "   Server Response: $($reader.ReadToEnd())" -ForegroundColor Yellow
    } catch {}
}

# ==========================================
# 3. Test Camera Latest (Verify Upload)
# ==========================================
Write-Host "`n--- 3. Testing Camera Latest ---"
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/camera/latest" -Method Get
    Write-Host "✅ Camera Latest Success: $($response | ConvertTo-Json -Depth 1)" -ForegroundColor Green
} catch {
    Write-Host "❌ Camera Latest Failed: $_" -ForegroundColor Red
}
