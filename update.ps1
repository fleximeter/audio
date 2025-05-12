# This is a modified Claude-generated script for updating the aus-analyzer package.
# You need to have the virtual environment activated before running it.
$DirectoryPath = "D:\Source\aus-analyzer\target\wheels"

# Validate that the directory exists
if (-not (Test-Path -Path $DirectoryPath -PathType Container)) {
    Write-Error "The specified directory does not exist: $DirectoryPath"
    exit 1
}

# Get the newest file in the directory
$newestFile = Get-ChildItem -Path $DirectoryPath | 
              Sort-Object LastWriteTime -Descending | 
              Select-Object -First 1

if ($newestFile) {
    pip uninstall aus-analyzer -y
    pip install $newestFile.FullName
} else {
    Write-Host "No files found in the specified directory."
}