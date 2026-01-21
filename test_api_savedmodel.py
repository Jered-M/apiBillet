import requests
import json
from pathlib import Path

test_image = Path('test_image.jpg')
if test_image.exists():
    with open(test_image, 'rb') as f:
        files = {'file': f}
        resp = requests.post('http://127.0.0.1:5000/predict', files=files, timeout=30)
        print('Status:', resp.status_code)
        if resp.status_code == 200:
            result = resp.json()
            print('✅ Top predictions:')
            for pred in result['predictions'][:5]:
                print(f"  {pred['class']}: {pred['confidence']*100:.1f}%")
            print('\n📊 All confidences for similar classes:')
            conf = result['all_confidences']
            print(f"  500_CDF: {conf['500_CDF']*100:.1f}%")
            print(f"  20_USD:  {conf['20_USD']*100:.1f}%")
else:
    print('Image not found')
