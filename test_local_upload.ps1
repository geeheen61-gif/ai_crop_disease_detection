
$baseUrl = "http://127.0.0.1:5000"

Write-Host "Testing Local Backend API at $baseUrl" -ForegroundColor Cyan

# 3. Test /camera/upload (Simulated)
# Create a dummy image file with valid JPEG header
$dummyImage = "dummy_test_local.jpg"
$jpegHeader = [byte[]](0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01)
Set-Content -Path $dummyImage -Value $jpegHeader -Encoding Byte
Write-Host "`n3. Testing /camera/upload (Simulated)..."
try {
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
Remove-Item $dummyImage
