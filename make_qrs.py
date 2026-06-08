import qrcode

# 5 benign URLs
benign = [
    "https://www.google.com",
    "https://en.wikipedia.org/wiki/QR_code",
    "https://www.python.org",
    "https://github.com",
    "https://www.bbc.com/news",
]

# 5 phishing-style URLs (fake patterns - do NOT visit)
malicious = [
    "http://paypal-verify-account.tk/login",
    "http://secure-bank-login.ml/update",
    "http://amaz0n-prize-claim.ga/win",
    "http://apple-id-verify.cf/signin",
    "http://faceb00k-security.xyz/reset",
]

for i, url in enumerate(benign):
    img = qrcode.make(url)
    img.save(f"physical_test/benign_{i}.png")
    print(f"saved benign_{i}.png")

for i, url in enumerate(malicious):
    img = qrcode.make(url)
    img.save(f"physical_test/malicious_{i}.png")
    print(f"saved malicious_{i}.png")

print("Done. Check the physical_test folder.")