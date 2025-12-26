
# train_features_and_svm.py
import os, torch, joblib
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
from glob import glob

# استخدم المسار الجديد بعد فك الضغط
DATA_DIR = "/content/animals10/raw-img"
SAVE_DIR = os.environ.get("SAVE_DIR", "/content/drive/MyDrive/species_project")
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

feat_model = models.mobilenet_v2(pretrained=True).features.to(device)
feat_model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def extract_feature(img_path):
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = feat_model(x)
        feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1,1)).squeeze().cpu().numpy()
    return feat

def load_dataset_from_folder(base_folder):
    X, y, labels = [], [], []
    class_dirs = sorted([d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder,d))])
    if not class_dirs:
        raise ValueError(f"No class subfolders found in {base_folder}")
    for idx, cls in enumerate(class_dirs):
        pth = os.path.join(base_folder, cls)
        files = glob(os.path.join(pth, "*"))
        print(f"Found {len(files)} images for class '{cls}'")
        for f in files:
            try:
                X.append(extract_feature(f))
                y.append(idx)
            except Exception as e:
                print("skip", f, e)
        labels.append(cls)
    return np.array(X), np.array(y), labels

if __name__ == "__main__":
    print("Loading from:", DATA_DIR)
    X, y, labels = load_dataset_from_folder(DATA_DIR)

    print("Features shape:", X.shape)
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True))
    clf.fit(X, y)

    y_pred = clf.predict(X)
    print("Train Accuracy:", classification_report(y, y_pred, target_names=labels))

    save_path = os.path.join(SAVE_DIR, "svc_pipeline.joblib")
    joblib.dump({'model': clf, 'labels': labels}, save_path)
    print("Saved model to:", save_path)
