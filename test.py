# quick_test.py (run from project root)
from backend.inference.models import predict_single_image
res = predict_single_image(r"D:\Infosys_AI-Tracefinder\Data\Wikipedia\HP\300\s11_107.tif", verbose=True)
print(res)
