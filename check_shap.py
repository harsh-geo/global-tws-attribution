import scipy.io as sio

data = sio.loadmat('outputs/tables/shap_results.mat', squeeze_me=True)
shap_results = data['shap_results']
for i, res in enumerate(shap_results):
    if res.size > 0:
        print(f"Basin {res['basin_id']}")
        try:
            print(f"Shape: {res['shap_values'].shape}")
        except Exception as e:
            print(f"Error checking shape: {e}")
            print(res['shap_values'])
