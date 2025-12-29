
$baseUrl = "https://ai-crop-disease-detection.onrender.com"

Write-Host "Testing Backend API at $baseUrl" -ForegroundColor Cyan

# 1. Test /sensors/store
$sensorData = @{
    temperature = 25.5
    humidity = 60.0
    soil = 1200
    rain = 4095
    light = 500.0
} | ConvertTo-Json

Write-Host "`n1. Testing /sensors/store..."
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/sensors/store" -Method Post -Body $sensorData -ContentType "application/json"
    Write-Host "   Success: $($response.ok)" -ForegroundColor Green
} catch {
    Write-Host "   Failed: $_" -ForegroundColor Red
}

# 2. Test /camera/latest
Write-Host "`n2. Testing /camera/latest..."
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/camera/latest" -Method Get
    Write-Host "   Latest Image URL: $($response.image_url)" -ForegroundColor Green
    if ($response.image_url) {
        Write-Host "   (URL is present)" -ForegroundColor Green
    } else {
        Write-Host "   (No image URL found - normal if no uploads yet)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   Failed: $_" -ForegroundColor Red
}

# 3. Test /camera/upload (Simulated)
# Create a VALID minimal JPEG using Python to satisfy Cloudinary validation
$dummyImage = "dummy_test.jpg"
$pythonScript = "import io, PIL.Image; img = PIL.Image.new('RGB', (10, 10), color = 'red'); img.save('$dummyImage')"
python -c "$pythonScript"

Write-Host "`n3. Testing /camera/upload (Simulated)..."
try {
    # Using raw binary upload (ESP32 style)
    $response = Invoke-RestMethod -Uri "$baseUrl/camera/upload" -Method Post -InFile $dummyImage -ContentType "image/jpeg"
    Write-Host "   Success: $($response.ok)" -ForegroundColor Green
    Write-Host "   URL: $($response.url)" -ForegroundColor Green
} catch {
    Write-Host "   Failed: $_" -ForegroundColor Red
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader $_.Exception.Response.GetResponseStream()
        Write-Host "   Response: $($reader.ReadToEnd())" -ForegroundColor Red
    }
}
if (Test-Path $dummyImage) { Remove-Item $dummyImage }
