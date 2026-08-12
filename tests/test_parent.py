
from pathlib import Path

print("file" + str(Path(__file__)))
print("file.parent" + str(Path(__file__).parent))
print("file.parent.parent" + str(Path(__file__).parent.parent))