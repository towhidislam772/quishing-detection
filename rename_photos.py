import os

photos_dir = r"C:\Users\USER\Desktop\QR paper\content\quishing-detection\physical_test\photos"

# Sorted alphabetically (matches WhatsApp's time-ordered listing)
files = sorted([f for f in os.listdir(photos_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

print(f"Found {len(files)} photos\n")
if len(files) != 30:
    print(f"WARNING: expected 30 photos, found {len(files)}")

conditions = ["normal", "angle", "dim"]

for i, old_name in enumerate(files):
    category = "benign" if i < 15 else "malicious"
    qr_idx   = (i % 15) // 3
    cond     = conditions[i % 3]
    new_name = f"{category}_{qr_idx}_{cond}.jpg"

    old_path = os.path.join(photos_dir, old_name)
    new_path = os.path.join(photos_dir, new_name)

    if os.path.exists(new_path) and old_path != new_path:
        print(f"SKIP (target exists): {new_name}")
        continue

    os.rename(old_path, new_path)
    print(f"{old_name}  ->  {new_name}")

print("\nDone.")
